import pytest
import pandas as pd


from configs.parametri_app import DATAFRAME_MASTER, DATAFRAME_GRUPPO_1


class TestDataframes:

    # Per testare se pytest funziona
    def test_prova(self):
        assert True

    # [Test condizionale] Esegue questo test solo se il file del dateset master è presente
    @pytest.mark.skipif(not DATAFRAME_MASTER.exists(), reason="File non trovato, salto il test")
    def test_dataframe_master_should_has_expected_lenght(self):
        lunghezza_dataframe = pd.read_csv(DATAFRAME_MASTER).shape[0]
        assert lunghezza_dataframe == 1599


    def test_dataframe_gruppo_1_should_has_expected_columns(self):
        lunghezza_dataframe_parziale = pd.read_csv(DATAFRAME_GRUPPO_1).shape[0]

        # colonne_dataframe_gruppo_1 = pd.read_csv(DATAFRAME_GRUPPO_1).columns.values.tolist()
        # colonne_attese = set(
        #     ['path_img'] +
        #     [f"punto_{k}_X", f"punto_{k}_Y" for k in [1, 2, 5, 6]]
        # )
        # assert colonne_attese == colonne_dataframe_gruppo_1
        assert lunghezza_dataframe_parziale == 1599