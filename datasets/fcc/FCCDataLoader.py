import numpy as np
import os
import matplotlib.pyplot as plt
import random
import linecache

class FCCDataLoader:
    def __init__(self, index_file, log_path=None, log_file=None, log_every=200, threshold=np.inf, msg=False, name="FCCDataLoader"):
        self.index_file_path = index_file
        self.log = []
        self.file_path = self.get_random_line(self.index_file_path).strip()
        self.count = 0
        self.max_count = sum(1 for line in open(self.file_path))
        self.log_path = log_path
        self.log_file = log_file
        self.log_every = log_every
        self.threshold = threshold
        self.yield_count = 0
        self.msg = msg
        self.name = name
        if self.log_path is not None:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.log_file = self.concatenate_path(self.log_path, self.log_file)
            # Clear the log file
            with open(self.log_file, 'w') as f:
                f.write('')
            self.log_count = 0

    def __iter__(self):
        return self

    @staticmethod 
    def concatenate_path(file_path, file_name):
        return os.path.join(file_path, file_name)
    
    @staticmethod
    def get_random_line(file_path):
        num_lines = sum(1 for line in open(file_path))
        random_line_number = random.randint(1, num_lines)
        return linecache.getline(file_path, random_line_number)

    def __next__(self):
        # Save log
        if self.log_file is not None:
            self.log_count += 1
            if self.log_count % self.log_every == 0:
                with open(self.log_file, 'a') as f:
                    for item in self.log:
                        f.write("%s\n" % item)
                self.log = []
        # Check stop condition
        if self.yield_count >= self.threshold:
            if self.log_file is not None and self.log:
                self.log_count += 1
                with open(self.log_file, 'a') as f:
                    for item in self.log:
                        f.write("%s\n" % item)
                self.log = []

            raise StopIteration
        
        # Load data
        if self.count < self.max_count:
            #print(self.file_path, self.count+1)
            #print(linecache.getline(self.file_path, self.count+1))
            thoroughput = int(linecache.getline(self.file_path, self.count+1).strip('\n'))
            self.log.append(thoroughput)
            self.count += 1
            self.yield_count += 1
            return thoroughput
        else:
            self.file_path = self.get_random_line(self.index_file_path).strip()
            self.count = 0
            self.max_count = sum(1 for line in open(self.file_path))
            thoroughput = int(linecache.getline(self.file_path, self.count+1).strip())
            self.log.append(thoroughput)
            self.count += 1
            self.yield_count += 1
            print(f"Loader {self.name} Switched to new trace: {self.file_path.split('/')[-1]}.")
            return thoroughput

    def tick(self):
        try:
            return next(self)
        except StopIteration:
            return None

class FCCRecoveryDataLoader:
    def __init__(self, recover_log_path):
        self.path = recover_log_path

    def __iter__(self):
        return map(int, (line for line in open(self.path)))
    
    def tick(self):
        try:
            return next(self)
        except StopIteration:
            return None


if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    file_name = '202301_cooked_cleaned.txt'
    file_path = os.path.join(os.path.dirname(current_file_path), file_name)
    loader = FCCDataLoader(file_path, log_path = os.path.join(os.path.dirname(current_file_path)), log_file="testlog", threshold=10000)
    count = 0

    print("Start generating items")

    record = list(loader)

    print(record[:5])
    

    print("Start retrieving items")

    retrieved = list(FCCRecoveryDataLoader(os.path.join(os.path.dirname(current_file_path), "testlog")))
    print(retrieved[:5])

    print(np.sum(np.array(record) == np.array(retrieved)))
        