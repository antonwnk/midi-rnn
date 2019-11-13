## Execution instructions (Adapted from https://github.com/brannondorsey/midi-rnn)

`midi-rnn` should work in MacOS and Linux environments. Open a terminal and run:
```bash
# clone this repo
git clone https://github.com/antonwnk/midi-rnn.git

# Install the dependencies. You may need to prepend sudo to 
# this command if you get an error
pip install -r requirements.txt
``` 

If you have CUDA installed and would like to train using your GPU, additionally run:
```bash
pip install tensorflow-gpu
``` 

## Training a Model

183 files from the [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/) are available inside the `data/midi` folder so that you can get started. Note that is basic RNN learns only from the monophonic tracks in MIDI files and simply ignores tracks that are observed to include polyphony.

Once you've got a collection of MIDI files you can train your model with `train.py` (note: for systems with a properly configured CuDNN installation - this includes Google Colab - the version `train-cudnn.py` is provided and offers a significant reduction in training time).

```bash
python train.py --data_dir data/midi
```

For a list of supported command line training flags, run:

```
python train.py --help
```

### Monitoring Training with Tensorboard

`model-rnn` logs training metrics using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). These logs are stored in a folder called `tensorboard-logs` inside of your `--experiment_dir`.

```
# Compare the training metrics of all of your experiments at once
tensorboard --logdir experiments/
```

Once Tensorboard is running, navigate your web browser to `http://localhost:6006` to view the training metrics for your model in real time.

## Generating MIDI

Once you've trained your model, you can generate MIDI files using `sample.py`.

```bash
python sample.py
```
or, for usage information:
```
python sample.py --help
```

By default, this creates 10 MIDI files using a model checkpoint from the most recent folder in `experiments/` and saves the generated files to `generated/` inside of that experiment directory (e.g. `experiments/01/generated/`). You can specify which model you would like to use when generating using the `--experiment_dir` flag. You can also specify where you would like to save the generated files by including a value for the `--save_dir` flag. For a complete list of command line flags, see below.

## How it works

This is a _very_ basic LSTM Recurrent Neural Network (RNN). It uses windows of 129-class one-hot encoded (0-127 = MIDI note numbers + 1 class to represent rests) as input for each step and creates a softmax probability distrobution over these 129 classes which it samples from to predict the next note in the sequence. That note is then appended to the window (poping the first note off the list to keep a fixed size window) and that window is then used as input for the prediction in the next time step. Many methods could be used to improve its performance (like for instance, using an encoder-decoder sequence-2-sequence model), however, `midi-rnn` should serve as a nice "naive" baseline to compare other machine learning MIDI generation tasks and algorithms against. 
