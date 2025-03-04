import os

def calculate_exemp_size(dataset, new_class, person):
    holdout_sizes = {
        'hhar': {
            ('3', '2'): {0: [100, 394, 1550, 6100, 120], 1: [100, 393, 1547, 6085, 120], 2: [100, 390, 1520, 5928, 120]},
            ('22', '21', '31'): {0: [100, 394, 1550, 6100, 180], 1: [100, 393, 1547, 6085, 180], 2: [100, 390, 1520, 5928, 180]},
            '111': {0: [100, 394, 1550, 6100, 240], 1: [100, 393, 1547, 6085, 240], 2: [100, 390, 1520, 5928, 240]},
            '1111': {0: [100, 394, 1550, 6100, 300], 1: [100, 393, 1547, 6085, 300], 2: [100, 390, 1520, 5928, 300]}
        },
        'motion': {
            ('3', '2'): {0: [100, 197, 390, 770, 120], 1: [100, 195, 379, 737, 120], 2: [100, 191, 366, 699, 120]},
            ('22', '21', '31'): {0: [100, 197, 390, 770, 180], 1: [100, 195, 379, 737, 180], 2: [100, 191, 366, 699, 180]},
            '111': {0: [100, 197, 390, 770, 240], 1: [100, 195, 379, 737, 240], 2: [100, 191, 366, 699, 240]},
            '1111': {0: [100, 197, 390, 770, 300], 1: [100, 195, 379, 737, 300], 2: [100, 191, 366, 699, 300]}
        },
        'uci': {
            ('3', '2'): {0: [100, 142, 201, 286, 120], 1: [100, 142, 201, 285, 120], 2: [100, 140, 196, 274, 120]},
            ('22', '21', '31'): {0: [100, 142, 201, 286, 180], 1: [100, 142, 201, 285, 180], 2: [100, 140, 196, 274, 180]},
            '111': {0: [100, 142, 201, 286, 240], 1: [100, 142, 201, 285, 240], 2: [100, 140, 196, 274, 240]},
        }, 
        'realworld': {
            ('3'): {0: [100, 303, 921, 2795, 120], 1: [100, 300, 898, 2692, 120], 2: [100, 299, 894, 2674, 120]},
            ('32','22'): {0: [100, 303, 921, 2795, 180], 1: [100, 300, 898, 2692, 180], 2: [100, 299, 894, 2674, 180]},
            ('211', '222'): {0: [100, 303, 921, 2795, 240], 1: [100, 300, 898, 2692, 240], 2: [100, 299, 894, 2674, 240]},
            ('1111', '2111', '3111'): {0: [100, 303, 921, 2795, 300], 1: [100, 898, 2692, 300], 2: [100, 299, 894, 2674, 300]},
        },
        'pamap': {
            '4': {0: [100, 240, 575, 1378, 120], 1: [100, 237, 562, 1332, 120], 2: [100, 235, 554, 1303, 120]},
            ('32','33'): {0: [100, 240, 575, 1378, 180], 1: [100, 237, 562, 1332, 180], 2: [100, 235, 554, 1303, 180]},
            ('321', '341'): {0: [100, 575, 1378, 240], 1: [100, 237, 562, 1332, 240], 2: [100, 235, 554, 1303, 240]},
            '2222': {0: [100, 240, 575, 1378, 300], 1: [100, 237, 562, 1332, 300], 2: [100, 235, 554, 1303, 300]},
            '11111': {0: [100, 240, 575, 1378, 360], 1: [100, 237, 562, 1332, 360], 2: [100, 235, 554, 1303, 360]},
        }
    }

    dataset_info = holdout_sizes.get(dataset, {})
    for classes, sizes_by_person in dataset_info.items():
        if new_class in classes if isinstance(classes, tuple) else new_class == classes:
            return sizes_by_person.get(person, None)
    return None

def get_output_file_paths(args, OUT_PATH, holdout_size):
    """
    Generates output file paths based on method and exemplar type.
    """

    # Define base folder structure
    base_path = f"{OUT_PATH}{args.dataset}/{args.base_classes}{args.new_classes}/Person_{args.person}"

    # Determine folder name based on method or exemplar type
    if args.exemplar == 'taskvae':
        folder_name = f"{base_path}/VAE_{args.vae_lat_sampling}_{args.latent_vec_filter}/log"
        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{args.vae_lat_sampling}_{args.latent_vec_filter}_{holdout_size}"

    elif args.method == 'kd_kldiv':
        folder_name = f"{base_path}/iCaRL_{args.number}/log"
        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{str(args.lamda_old)}_{holdout_size}"

    elif args.method == 'cn_lfc_mr':
        folder_name = f"{base_path}/LUCIR_{args.number}/log"
        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{str(args.lamda_base)}_{holdout_size}"

    elif args.method == 'ce_ewc':
        folder_name = f"{base_path}/EWC_Replay_{args.number}/log"
        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{holdout_size}"

    else:
        folder_name = f"{base_path}/Random_{args.number}/log"
        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{holdout_size}"

    # Create folders for log and statistics
    log_folder = os.path.join(folder_name, "log")
    os.makedirs(log_folder, exist_ok=True)

    # Define output file paths
    outfile = f"{log_folder}/{filename_suffix}.txt"

    return outfile
