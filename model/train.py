import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import *


def pairwise_cosine_contrastive_loss(attn_outputs, aspect_embs, aspect_labels, margin=0.2):
    attn_all = torch.stack(attn_outputs, dim=1)     
    aspect_all = torch.stack(aspect_embs, dim=1)    
    
    dim = attn_all.shape[-1]
    attn_flat = attn_all.reshape(-1, dim)           
    aspect_flat = aspect_all.reshape(-1, dim)       
    
    labels_01 = aspect_labels.reshape(-1).float()
    target = labels_01 * 2 - 1.0

    loss = F.cosine_embedding_loss(
        attn_flat, 
        aspect_flat, 
        target, 
        margin=margin, 
        reduction='mean' 
    )

    return loss

class MultiTaskModel(pl.LightningModule):
    def __init__(self, num_aspects, encoder, lambda_contrastive, global_class_weights=None, learning_rate=2e-5, topk_layer=4):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.num_aspects = num_aspects
        self.topk_layer = topk_layer
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.lambda_contrastive = lambda_contrastive

        if global_class_weights is not None:
            self.register_buffer("global_weights", torch.tensor(global_class_weights, dtype=torch.float32))
        else:
            self.global_weights = None

        self.base_hidden_size = self.encoder.config.hidden_size
        self.hidden_size = self.base_hidden_size * topk_layer
        self.hidden_dim = 512

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)
        self.aspect_proj = nn.Linear(self.base_hidden_size, self.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(0.3)
        )
        self.aspect_detector = nn.Linear(self.hidden_dim * 2, 1)
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_dim * 2, 4) for _ in range(num_aspects)
        ])
        self.global_classifier = nn.Linear(self.hidden_dim * 2, 4)

        self.dropout = nn.Dropout(0.3)
        self.aspect_loss_fn = AsymmetricLossOptimized()
        self.sentiment_loss_fn = AsymmetricLossOptimized()
        self.global_loss_fn = nn.CrossEntropyLoss(weight=self.global_weights)
        # self.global_loss_fn = FocalLoss()

        self.train_accuracies = []
        self.val_accuracies = []
        self.val_losses = [] 

    def forward(self, x):
        input_ids, attention_mask, auxiliary_embeddings = x

        if "bartpho" in self.encoder.config._name_or_path.lower():
            outputs = self.encoder.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states
        topk_layer = min(self.topk_layer, len(hidden_states))
        text_hidden = torch.cat([hidden_states[-i] for i in range(1, topk_layer + 1)], dim=-1)

        aspect_presence_preds, sentiment_preds = [], []
        attn_outputs_all, aspect_embs_proj = [], []
        all_feats = []

        for i in range(self.num_aspects):
            aspect_emb = auxiliary_embeddings[:, i, :]
            aspect_emb = self.aspect_proj(aspect_emb)
            aspect_embs_proj.append(aspect_emb)

            aspect_query = aspect_emb.unsqueeze(1)
            attn_output, _ = self.multihead_attn(query=aspect_query, key=text_hidden, value=text_hidden)
            attn_outputs_all.append(attn_output.squeeze(1))

            interaction = attn_output.squeeze(1) * aspect_emb
            concat = torch.cat([attn_output.squeeze(1), aspect_emb, interaction], dim=-1)
            feat = self.proj(concat)
            all_feats.append(feat)

            presence_logit = self.aspect_detector(feat).squeeze(-1)
            sentiment_logit = self.aspect_classifiers[i](feat)

            aspect_presence_preds.append(presence_logit)
            sentiment_preds.append(sentiment_logit)

        aspect_presence = torch.stack(aspect_presence_preds, dim=1) # [bs, num_aspects]
        aspect_presence = torch.sigmoid(aspect_presence)  # Apply sigmoid to get probabilities
        aspect_sentiment = torch.stack(sentiment_preds, dim=1) # [bs, num_aspects, 4]
        global_feat = torch.mean(torch.stack(all_feats, dim=1), dim=1)
        global_logits = self.global_classifier(global_feat) # [bs, 4]

        return aspect_presence, aspect_sentiment, attn_outputs_all, aspect_embs_proj, global_logits
    
    def calculate_metrics(self, aspect_presence, sentiment_preds, global_logits, batch):
        metrics = {}
        
        aspect_preds_binary = (aspect_presence > 0.5).float()
        metrics['aspect_acc'] = (aspect_preds_binary == batch['aspects']).float().mean()

        sentiment_preds_classes = torch.argmax(sentiment_preds, dim=-1)
        sentiment_targets = torch.argmax(batch['labels'], dim=-1)
        mask = (batch['aspects'] == 1)
        
        if mask.sum() > 0:
            correct = (sentiment_preds_classes[mask] == sentiment_targets[mask]).float()
            metrics['sentiment_acc'] = correct.mean()
        else:
            metrics['sentiment_acc'] = torch.tensor(0.0, device=self.device)

        global_preds_classes = torch.argmax(global_logits, dim=-1)
        if batch['global_label'].dim() > 1:
            global_targets = torch.argmax(batch['global_label'], dim=-1)
        else:
            global_targets = batch['global_label']
            
        metrics['global_acc'] = (global_preds_classes == global_targets).float().mean()

        return metrics

    def compute_loss(self, batch, outputs):
        aspect_presence, sentiment_preds, attn_outputs_all, aspect_embs_proj, global_logits = outputs

        sentiment_loss = self.sentiment_loss_fn(sentiment_preds, batch['labels'])
        aspect_loss = self.aspect_loss_fn(aspect_presence, batch['aspects'])
        global_loss = self.global_loss_fn(global_logits, batch['global_label'])

        contrastive_loss = pairwise_cosine_contrastive_loss(attn_outputs_all, aspect_embs_proj, batch['aspects'])

        total_loss = sentiment_loss + aspect_loss + global_loss + self.lambda_contrastive * contrastive_loss

        self.log("sentiment_loss", sentiment_loss, prog_bar=False)
        self.log("aspect_loss", aspect_loss, prog_bar=False)
        self.log("global_loss", global_loss, prog_bar=False)
        self.log("contrastive_loss", contrastive_loss, prog_bar=False)

        return total_loss, sentiment_preds

    def training_step(self, batch, _):
        outputs = self((batch['input_ids'], batch['attention_mask'], batch['auxiliary_embeddings']))
        loss, sentiment_preds = self.compute_loss(batch, outputs)

        aspect_presence, _, _, _, global_logits = outputs

        metrics = self.calculate_metrics(aspect_presence, sentiment_preds, global_logits, batch)

        self.train_accuracies.append(metrics['sentiment_acc'])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_aspect_acc", metrics['aspect_acc'], prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_sentiment_acc", metrics['sentiment_acc'], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_global_acc", metrics['global_acc'], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, _):
        outputs = self((batch['input_ids'], batch['attention_mask'], batch['auxiliary_embeddings']))
        loss, sentiment_preds = self.compute_loss(batch, outputs)

        aspect_presence, _, _, _, global_logits = outputs

        metrics = self.calculate_metrics(aspect_presence, sentiment_preds, global_logits, batch)

        self.val_accuracies.append(metrics['sentiment_acc'])
        self.val_losses.append(loss)
        self.log("val_loss_batch", loss, prog_bar=False)
        self.log("val_sentiment_acc_batch", metrics['sentiment_acc'], prog_bar=True)
        self.log("val_aspect_acc_batch", metrics['aspect_acc'], prog_bar=True)
        self.log("val_global_acc_batch", metrics['global_acc'], prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        if self.train_accuracies:
            avg_acc = torch.stack(self.train_accuracies).mean()
            self.log("train_acc_epoch", avg_acc, prog_bar=True)
            self.train_accuracies.clear()

    def on_validation_epoch_end(self):
        if self.val_accuracies:
            avg_acc = torch.stack(self.val_accuracies).mean()
            self.log("val_acc_epoch", avg_acc, prog_bar=True)
            self.val_accuracies.clear()

        if self.val_losses:
            avg_loss = torch.stack(self.val_losses).mean()
            self.log("val_loss", avg_loss, prog_bar=True)
            self.val_losses.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
