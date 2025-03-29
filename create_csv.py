import os
import csv
import random

path = "./poisonedDataset"
dir_list = os.listdir(path)
 
data = [['Path', 'Perturbed']]

for d in dir_list:
    poisoned = "Yes" if d.endswith(".pt") else "No"
    data.append([d, poisoned])
    
rows = data[1:]
random.shuffle(rows)
rows.insert(0, ['Path', 'Perturbed'])
            
with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)