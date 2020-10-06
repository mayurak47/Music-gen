# Music-gen
An attempt to generate classical piano music using 1D CNNs, as opposed to sequence models or generative models.

## Dataset 
I used [this](https://www.kaggle.com/soumikrakshit/classical-music-midi) classical music dataset, consisting of piano pieces by Beethoven, Mozart, Chopin, Liszt and other classical composers.

## Motivation
LSTMs take extremely long to train, especially when stacked. An LSTM model running on Google Colab took about 20 hours to train on this dataset, and the output consisted of very few distinct notes, leading to a very monotonous and artificial sounding piece. Since the dataset isn't very large, GANs might not be a good choice.

## Data Preparation
Each midi file was split into its constituent notes or chords, and a dictionary mapped each note/chord to an integer. Music generation was cast as a classification problem, where the model outputs the next note/chord, given a fixed length sequence of notes/chords. 

## Model 
The model consists of multiple Conv1D layers with gradually decreasing kernel sizes, with a couple of MaxPool1D layers in between. The output is further processed using fully connected layers, which then output the probabilities for the next note or chord. Categorical crossentropy is used as the loss function, with an Adam optimizer.

## Results
A sequence of notes from the validation set was provided as input. [Here](https://gofile.io/d/JxLsdC) is a sample output. While the audio sounds much more dynamic and uses a wider range of notes than the LSTM model's output, it is obvious that the model has trouble capturing discernible long-term dependencies.

## Future work
It might be worth trying a hybrid CNN-LSTM model, exploiting the advantages of both models. The features learnt by the convolutional layers could be run through LSTM units.

## References
[How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
