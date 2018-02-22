import numpy as np
import pickle

# Determine the files needed for processing
FILE_LIST = ["aircraftCarrier.npy"]
NUM_FILES = len(FILE_LIST)

# Limit the number of images processed
TOTAL_LIMIT = 10000
# Ratio of testing images to training
TEST_RATIO = 0.5

# Gen a random seed
SEED = np.random.randint(1, 10e6)
print SEED
np.random.seed(SEED)

# Initialize Output Sets
TRAIN_IMAGES = []
TRAIN_LABELS = []
TEST_IMAGES = []
TEST_LABELS = []

# Intermediary Output Set
TRAIN = []
TEST = []

for F in range(NUM_FILES):
    FILE = FILE_LIST[F]
    dataset = np.load(FILE)
    dataset = dataset[:int(TOTAL_LIMIT/NUM_FILES)]
    test_items = [[item, F] for item in dataset[:int(TEST_RATIO*len(dataset))]]
    train_items = [[item, F] for item in dataset[int(TEST_RATIO*len(dataset)):]]
    TEST.extend(test_items)
    TRAIN.extend(train_items)

np.random.shuffle(TEST);
np.random.shuffle(TRAIN);

# Split the shuffled data
for item in TRAIN:
    TRAIN_IMAGES.append(item[0])
    TRAIN_LABELS.append(item[1])

for item in TEST:
    TEST_IMAGES.append(item[0])
    TEST_LABELS.append(item[1])


# Save the files as .npy files (for use in algo)
np.save("QD_TRAIN_IMAGES.npy", np.array(TRAIN_IMAGES));
np.save("QD_TRAIN_LABELS.npy", np.array(TRAIN_LABELS));
np.save("QD_TEST_IMAGES.npy", np.array(TRAIN_IMAGES));
np.save("QD_TESTLABELS.npy", np.array(TEST_LABELS));

        
        
