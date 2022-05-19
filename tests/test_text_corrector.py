import imp
from src.main import TextCorrectorPL


text_corrector = TextCorrectorPL()

def test_dummy():
    assert 1 == 1


def test_is_in_morfeusz_dict():
    assert text_corrector._TextCorrectorPL___is_in_morfeusz_dict('słowo') == True
    assert text_corrector._TextCorrectorPL___is_in_morfeusz_dict('slowo') == False

def test_find_freq():
    assert isinstance(text_corrector._TextCorrectorPL__find_freq('słowo'),int)
    assert text_corrector._TextCorrectorPL__find_freq('asd232rfs') == 0

def test_diacritic_combinations():
    assert len(text_corrector._TextCorrectorPL__diacritic_combinations('asc')) == 8
    assert "żółw" in text_corrector._TextCorrectorPL__diacritic_combinations('zolw')

def test_remove_one_letter_from_word():
    assert 'sowo' in text_corrector._TextCorrectorPL__remove_one_letter_from_word('słowo')

def test_swap_two_adjacent_letters():
    assert 'łsowo' in text_corrector._TextCorrectorPL__swap_two_adjacent_letters('słowo')

def test_qwerty_keyboard_typos():
    assert 'słiwo' in text_corrector._TextCorrectorPL__qwerty_keyboard_typos('słowo')

