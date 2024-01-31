import datetime
from tqdm import tqdm

def init_seed(seed: int):
    import torch
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        
def get_local_time_in_taiwan() -> datetime.datetime:
    utc_now = datetime.datetime.utcnow()
    taiwan_now = utc_now + datetime.timedelta(hours=8) # Taiwan in UTC+8
    return taiwan_now

def get_progress_bar(identifer: str, total_steps: int) -> tqdm:
    """Get the progress bar
    """
    progress_bar = tqdm(total = total_steps, 
                        desc = "{:10s}".format(identifer), 
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    return progress_bar