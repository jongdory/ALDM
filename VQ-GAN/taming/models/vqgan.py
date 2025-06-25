import torch
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.models.normalization import SPADEGenerator
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 num_classes=None, # list of modalities to use
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 stage=1,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.spade = SPADEGenerator(num_classes, ddconfig["z_channels"])
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
            
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, y=None):
        quant, diff, _ = self.encode(input)
        if y is not None: 
            quant = self.spade(quant, y)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, "source")
        targets = self.get_input(batch, "target")
        

        if self.stage == 1: 
            y = None
            xrec, qloss = self(inputs)
        else:
            y = batch["target_class"].long()
            xsrc, qloss, _ = self.encode(inputs)
            xrec = self.spade(xsrc, y)
            inputs, _, _ = self.encode(targets)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, inputs, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), label=y,split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, inputs, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), label=y, split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, "source")
        targets = self.get_input(batch, "target")

        if self.stage == 1: 
            y = None
            xrec, qloss = self(inputs)
        else:
            y = batch["target_class"].long()
            xsrc, qloss, _ = self.encode(inputs)
            xrec = self.spade(xsrc, y)
            inputs, _, _ = self.encode(targets)

        aeloss, log_dict_ae = self.loss(qloss, inputs, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, inputs, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.stage == 1:
            for p in self.spade.parameters(): p.requires_grad = False
            for p in self.encoder.parameters(): p.requires_grad = True
            for p in self.decoder.parameters(): p.requires_grad = True
            opt_ae = torch.optim.Adam(
                                  list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        else:
            for p in self.spade.parameters(): p.requires_grad = True
            for p in self.encoder.parameters(): p.requires_grad = False
            for p in self.decoder.parameters(): p.requires_grad = False
            opt_ae = torch.optim.Adam(list(self.spade.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        inputs = self.get_input(batch, "source").to(self.device)
        targets = self.get_input(batch, "target").to(self.device)
        y = batch["target_class"].long()
        if self.stage == 1:
            xrec, _ = self(inputs)
            log["source"] = inputs
            log[f"recon"] = xrec
        else:
            xsrc, _ = self(inputs)
            xrec, _ = self(inputs, y)
            log["source"] = inputs
            log["target"] = targets
            log["source_recon"] = xsrc
            log["target_spade"] = xrec
        return log

