# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer=dict(
        type='AdamW',
        lr=0.0002,
        weight_decay=0.05,
        eps=1e-6,
        betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-5)
  
runner = dict(type='EpochBasedRunner', max_epochs=72)
checkpoint_config = dict(interval=1)
