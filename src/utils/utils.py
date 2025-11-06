from configs.parametri_app import * # parametri dell'applicazione da usare in giro per il progetto


'''
Modulo in cui inserisco funzioni di uso comune e di utilità varie.
'''

def get_num_files(estensione : str) -> int:
    '''
    Restituisce il numero di file con quella determinata estensione presenti dentro la lista
    di tutti i file (jpg, txt) del dataset.

    Args:
        estensione: può essere 'jpg' o 'txt'.
    '''
    return len(list(filter( lambda nome_file: (f"{estensione.lower()}" in nome_file), FILES )))

ok = '[✓]'
fail = '[x]'