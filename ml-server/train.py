import os
import torch
from torchvision import  transforms
from datasets import load_dataset
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
from transformers import TrainingArguments
from transformers import Trainer
from config import CFG

# Load dataset
model_name = CFG.base_model
data_path = CFG.data_path
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

dataset = load_dataset("imagefolder", data_files={"train": os.path.join(data_path,"Training/**"),\
                                                  "test": os.path.join(data_path,"Testing/**") ,\
                                                  "valid": os.path.join(data_path,"Validation/**")})
train_ds = dataset['train']
val_ds = dataset['valid']
test_ds = dataset['test']

#Specify transforms
normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
resize = tuple(feature_extractor.size.values())
_train_transforms = transforms.Compose([transforms.Resize(resize),                          
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(), normalize
                                       ])
_val_transforms = transforms.Compose([transforms.Resize(resize),                          
                                       transforms.ToTensor(), normalize
                                       ])

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image) for image in examples['image']]
    return examples

train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels = train_ds.features['label'].names
labels = [i.lower() for i in labels]

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
# Train model
training_args = TrainingArguments(
  output_dir=CFG.save_path,
  per_device_train_batch_size=CFG.batch_size,
  evaluation_strategy="steps",
  num_train_epochs=CFG.epohcs,
  fp16=CFG.fp16,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=CFG.learning_rate,
  save_total_limit=2,
  remove_unused_columns=False,
  report_to='none',
  push_to_hub=False,
  load_best_model_at_end=True,
  
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_model()

metrics = trainer.evaluate(test_ds)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# Calculate metrics

# Test model

# Save model
