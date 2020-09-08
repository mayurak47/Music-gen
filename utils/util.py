from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
import torch

def get_notes():
    """ Parse midi files and return list of songs """
    all_songs = []

    for file in glob.glob("data/*/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        curr_song_notes = []

        notes_to_parse = None

        try: 
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: 
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                curr_song_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                curr_song_notes.append('.'.join(str(n) for n in element.normalOrder))
        all_songs.append(curr_song_notes)

    return all_songs

def prep_sequences(all_songs, sequence_length=100):
    """ Prepare the sequences used by the Neural Network """
    # get all pitch names
    vocab_set = set()
    for song in all_songs:
        for note in song:
            vocab_set.add(note)
    
    pitchnames = sorted(vocab_set)
    n_vocab = len(vocab_set)
     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for song in all_songs:
        for i in range(len(song)-sequence_length):
            sequence_in = song[i:i + sequence_length]
            sequence_out = song[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    return (network_input, network_output)

def generate_midi(model, dataset, vocab_set, output_filename="output.mid"):
    start = np.random.randint(0, len(dataset)-1)
    pitchnames = sorted(vocab_set)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = dataset[start][0].squeeze()
    prediction_output = []

    # generate 500 notes
    with torch.no_grad():
        for _ in range(500):
            prediction_input = pattern.reshape(-1, 1, 100)
            prediction = model(prediction_input.cuda())
            
            index = torch.max(prediction, 1)[1].item()
            result = int_to_note[index]
            prediction_output.append(result)

            new_note = torch.tensor(index/len(s)).view(1)    
            pattern = torch.cat((pattern, new_note))
            pattern = pattern[1:len(pattern)]

    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp="output/" + str(output_filename))