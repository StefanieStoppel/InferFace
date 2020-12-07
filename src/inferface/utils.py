import csv
import os


def walk_dir_in_batches(directory, batch_size=128):
    walk_dirs_generator = os.walk(directory)
    for dirname, subdirectories, filenames in walk_dirs_generator:
        for i in range(0, len(filenames), batch_size):
            yield [os.path.join(dirname, filename) for filename in filenames[i:i + batch_size]]


def write_to_csv(row_data, csv_file_name):
    with open(csv_file_name, 'w', newline='') as csv_file:
        embedding_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
        embedding_writer.writerows(row_data)
