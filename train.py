import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_voc_instances

# データの登録
DatasetCatalog.register('my_dataset', lambda: load_voc_instances('train_data', 'train', ['target', 'green']))
MetadataCatalog.get('my_dataset').thing_classes = ['target', 'green']

# モデルの取得
model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
cfg = model_zoo.get_config(model, trained=True)

# 訓練データをセット
cfg.DATASETS.TRAIN = ('my_dataset',)
# テストデータをセット（今回はセットしない）
cfg.DATASETS.TEST = () 

cfg.DATALOADER.NUM_WORKERS = 4                  # default:4
cfg.SOLVER.IMS_PER_BATCH = 2                    # default:16
cfg.SOLVER.BASE_LR = 0.00025                    # dafault:0.02
cfg.SOLVER.MAX_ITER = 1000                      # default:270000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # dafault:512 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2             # dafault:80

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.train()