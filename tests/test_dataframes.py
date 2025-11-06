import pytest
import pandas as pd


from configs.parametri_app import DATAFRAME_MASTER


class TestDataframes:

    # Per testare se pytest funziona
    def test_prova(self):
        assert True

    # Fai questo test solo se il file del dateset master è presente
    @pytest.mark.skipif(not DATAFRAME_MASTER.exists(), reason="File non trovato, salto il test")
    def test_dataframe_master_should_has_expected_lenght(self):
        lunghezza_dataframe = pd.read_csv(DATAFRAME_MASTER).shape[0]
        assert lunghezza_dataframe == 1599
