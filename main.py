####################################   Load the data from file. ####################################
import read_data

train_tokens, train_tags = read_data.read_data('data/train.txt')
validation_tokens, validation_tags = read_data.read_data('data/validation.txt')
test_tokens, test_tags = read_data.read_data('data/test.txt')


#Print the first three to check the current result
for i in range(3):
    for token, tag in zip(train_tokens[i], train_tags[i]):
        print('%s\t%s' % (token, tag))
    print()


####################################  Prepare dictionaries ####################################
import prepare_dictionaries
