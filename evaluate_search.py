import features
import matplotlib.pyplot as plt
import numpy as np

features.load_features()
images = [
        "veggie-images/test/Bean/0001.jpg",
        "veggie-images/test/Bitter_Gourd/1201.jpg",
        "veggie-images/test/Bottle_Gourd/1001.jpg",
        "veggie-images/test/Brinjal/0871.jpg",
        "veggie-images/test/Broccoli/1001.jpg",
        "veggie-images/test/Cabbage/0929.jpg",
        "veggie-images/test/Capsicum/1001.jpg",
        "veggie-images/test/Carrot/1001.jpg",
        "veggie-images/test/Cauliflower/1048.jpg",
        "veggie-images/test/Cucumber/1001.jpg",
        "veggie-images/test/Papaya/1198.jpg",
        "veggie-images/test/Potato/1001.jpg",
        "veggie-images/test/Pumpkin/1001.jpg",
        "veggie-images/test/Radish/1001.jpg",
        "veggie-images/test/Tomato/1001.jpg",
        ]
print(len(images))

acc = {}
for model in features.models:
    acc[model] = 0
for image in images:
    cathegory = image.split("/")[-2]
    print(cathegory)
    for model in features.models:
        order, similarity = features.get_image_order(image, model)
        correct = 0
        for o in order[:200]:
            c = features.model_imagenames[model][o].split("/")[-2]
            if c == cathegory:
                correct += 1
        acc[model] +=  correct/200

for m, a in acc.items():
    acc[m]=a/15


models = list(acc.keys())
print(models)
models = ["Xception", "EfficientNetB0", "Custom Network", "ResNet50"]
accuracy = list(acc.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color='skyblue')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Model Name', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Model Search Accuracy Comparison', fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(rotation=45, ha='right')

plt.tight_layout()

plt.savefig("accuracy.png", dpi=300)
