# -*- coding: utf-8 -*-
"""final AI builder 2024.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19BT_QvQ_0GQo1Rti0SMg292Ogfm9VHFe

#final

##test
"""

!pip install datasets
!pip install evaluate
!pip install transformers[torch]

from google.colab import drive
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

drive.mount('/content/drive')

os.listdir("/content/drive/MyDrive/train_AI/new_structure")

"""## Dataset Loading"""

import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Image
from datasets import Dataset

seed_num = 112233

# โหลดชุดข้อมูล
data_root_path = "/content/drive/MyDrive/train_AI/new_structure"
class_list = os.listdir(data_root_path)

df_total = pd.DataFrame()
class_list = os.listdir(data_root_path)
for class_name in class_list:
  data_temp_pd = pd.DataFrame()
  if class_name == 'cataract':
    file_path = os.path.join(data_root_path, class_name, 'class1')
  else:
    file_path = os.path.join(data_root_path, class_name)

  file_list = os.listdir(file_path)
  file_list = [file_name for file_name in file_list if file_name.endswith(".jpg")]
  full_file_path = [os.path.join(file_path,file_name) for file_name in file_list]
  data_temp_pd['id'] = file_list
  data_temp_pd['image'] = full_file_path
  data_temp_pd['label'] = class_name
  df_total = pd.concat([df_total, data_temp_pd])

df_total['label'] = df_total['label'].str.lower()

df_total['label'].value_counts()

id2label = df_total[['label']].drop_duplicates().reset_index(drop = True)['label'].to_dict()
label2id  = {key:value for value, key in id2label.items()}
df_total['label'] = df_total['label'].map(label2id)

df_train, df_test = train_test_split(df_total, test_size=0.05, random_state=seed_num, stratify= df_total['label'])

label2id

# Change from pandas dataframe to be datasets object
dataset_train = Dataset.from_dict(df_train)
dataset_train = dataset_train.cast_column("image", Image())

dataset_test = Dataset.from_dict(df_test)
dataset_test = dataset_test.cast_column("image", Image())

"""## Image preprocessor"""

from transformers import AutoImageProcessor, DefaultDataCollator
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, Resize, ToTensor, Lambda, RandomHorizontalFlip
import io

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None

elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
val_transforms = Compose([Resize(size), CenterCrop(crop_size),ToTensor(), normalize])

def transforms_train(examples):
    examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

def transforms_validate(examples):
    examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

dataset_train = dataset_train.with_transform(transforms_train)
dataset_test = dataset_test.with_transform(transforms_validate)

"""## Performance Metrics Functions"""

 
"""### Hugging Face Trainer"""

from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id.keys()),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir = "vit-base-classification-Eye-Diseases-New02",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate= 7e-5,            # Main value for fine tuning
    per_device_train_batch_size=30,# Main value for fine tuning
    gradient_accumulation_steps=2, # Main value for fine tuning
    per_device_eval_batch_size=50, # Main value for fine tuning
    num_train_epochs=30,           # Main value for fine tuning, starting from low number !!
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

"""## Train !"""

train_results = trainer.train()
results = trainer.evaluate()
print(results)

print(trainer.state.best_model_checkpoint)