datasets = []
during_dataset = False
DATASET_COLUMN = 'DatasetName'

subsets = []
during_subset = False
SUBSET_COLUMN = 'SubsetName'

sfts = []
during_sft = False
ISSFT_COLUMN = 'IsSFT'

print(f"Paste the '{DATASET_COLUMN}' column")

while True:

    dataset = input()

    if dataset == "END_OF_DATASET":
        datasets.append(dataset)
        with open("datasets.txt", "w") as f:
            f.write("\n".join(datasets))
        print(f"'{DATASET_COLUMN}' column saved. Then you can paste the '{SUBSET_COLUMN}' column.")

    elif dataset == "END_OF_SUBSET":
        subsets.append(dataset)
        with open("subsets.txt", "w") as f:
            f.write("\n".join(subsets))
        print(f"'{SUBSET_COLUMN}' column saved. Then you can paste the '{ISSFT_COLUMN}' column.")

    elif dataset == "END_OF_SFT":
        sfts.append(dataset)
        with open("sfts.txt", "w") as f:
            f.write("\n".join(sfts))
        print("'{ISSFT_COLUMN}' column saved. then you can press Ctrl+C to exit")

    elif dataset == DATASET_COLUMN:
        during_dataset = True
        continue

    elif dataset == SUBSET_COLUMN:
        during_subset = True
        continue

    elif dataset == ISSFT_COLUMN:
        during_sft = True
        continue

    if during_dataset:
        datasets.append(dataset.strip())

    if during_subset:
        subsets.append(dataset.strip())

    if during_sft:
        sfts.append(dataset.strip())
