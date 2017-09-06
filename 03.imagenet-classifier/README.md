# ImageNET and MobileNET image classifier

# Description
This is an implementation of an algorithm which uses MobileNET or ImageNET in order to classify images.
The model can also be trained to classify new user-defined labels

# Parameters
In order to change your preferred classifier, edit your environmental variables.
They are found in your ```.bash_profile``` file, in your local user directory.
Or you can just put it as a parameter

# TensorFlow: Mobile net settings
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

# Commands
### Tensorboard: Run
tensorboard --logdir tf_files/training_summaries &

### Hyperparameters
```--learning_rate``` - Training speed
0.01    - Default
0.005   - Better accurancy, slower training speed
1       - Better training speed

```--summaries_dir``` - Setting separate summary folders, in order to be displayed in TensorBoard

### Training: Retrain using Transfer learning via the provided examples in ```image_dir```
```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"mobilenet_0.50_224" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="mobilenet_0.50_224" \
  --image_dir=tf_files/plants
```

### Detect: Label a class
```
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```