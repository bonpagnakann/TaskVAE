
params_map = {
    'hhar': {
        '22': {'kd_kldiv': '0.1', 'cn_lfc_mr': '0.5'},
        '1111': {'kd_kldiv': '0.3', 'cn_lfc_mr': '1.0'},
        '31': {'kd_kldiv': '0.1', 'cn_lfc_mr': '0.5'},
        '111': {'kd_kldiv': '0.3', 'cn_lfc_mr': '0.5'},
        '21': {'kd_kldiv': '0.1', 'cn_lfc_mr': '0.5'},
        '3': {'kd_kldiv': '0.1', 'cn_lfc_mr': '0.3'},
        '2': {'kd_kldiv': '0.3', 'cn_lfc_mr': '0.8'}
    },
    'motion': {
        '22': {'kd_kldiv': '1.0', 'cn_lfc_mr': '0.3'},
        '1111': {'kd_kldiv': '0.3', 'cn_lfc_mr': '0.5'},
        '31': {'kd_kldiv': '0.3', 'cn_lfc_mr': '0.5'},
        '111': {'kd_kldiv': '0.5', 'cn_lfc_mr': '0.3'},
        '21': {'kd_kldiv': '0.5', 'cn_lfc_mr': '0.5'},
        '3': {'kd_kldiv': '0.3', 'cn_lfc_mr': '0.5'},
        '2': {'kd_kldiv': '0.3', 'cn_lfc_mr': '0.5'}
    },
    'uci': {
        '22': {'kd_kldiv': '2.0', 'cn_lfc_mr': '0.3'},
        '31': {'kd_kldiv': '2.0', 'cn_lfc_mr': '0.3'},
        '111': {'kd_kldiv': '2.0', 'cn_lfc_mr': '0.3'},
        '21': {'kd_kldiv': '2.0', 'cn_lfc_mr': '0.3'},
        '3': {'kd_kldiv': '2.0', 'cn_lfc_mr': '0.3'},
        '2': {'kd_kldiv': '2.0', 'cn_lfc_mr': '0.3'}
    },
    'pamap': {
        '2222': {'kd_kldiv': '1.0', 'cn_lfc_mr': '0.3'},
        '11111': {'kd_kldiv': '1.0', 'cn_lfc_mr': '1.0'},
        '341': {'kd_kldiv': '0.5', 'cn_lfc_mr': '0.5'},
        '321': {'kd_kldiv': '0.1', 'cn_lfc_mr': '0.8'},
        '32': {'kd_kldiv': '0.1', 'cn_lfc_mr': '1.0'},
        '4': {'kd_kldiv': '0.1', 'cn_lfc_mr': '1.0'}
    },
    'realworld': {
        '222': {'kd_kldiv': '1.0', 'cn_lfc_mr': '1.5'},
        '211': {'kd_kldiv': '1.0', 'cn_lfc_mr': '2.0'},
        '3111': {'kd_kldiv': '0.5', 'cn_lfc_mr': '2.0'},
        '2111': {'kd_kldiv': '1.0', 'cn_lfc_mr': '2.0'},
        '1111': {'kd_kldiv': '1.0', 'cn_lfc_mr': '1.0'},
        '3': {'kd_kldiv': '0.3', 'cn_lfc_mr': '2.5'}
    }
}

def get_default_params(dataset, scenario, method):
    """
    Retrieve `lamda_old` and `lamda_base` values based on dataset, scenario, and method.
    """
    dataset = dataset.lower()  # Normalize dataset name
    scenario = str(scenario)  # Convert scenario to string

    lamda_value = float(params_map[dataset][scenario][method])
    
    return lamda_value, lamda_value  # Use the same value for both lamda_old and lamda_base

