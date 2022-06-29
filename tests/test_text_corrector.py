# -*- coding: utf-8 -*-
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
    assert "zdzblo" in text_corrector._TextCorrectorPL__diacritic_combinations('zdzblo')
    assert "zdzblo" in text_corrector._TextCorrectorPL__diacritic_combinations('źdźbło')

def test_remove_one_letter_from_word():
    assert 'sowo' in text_corrector._TextCorrectorPL__remove_one_letter_from_token('słowo')

def test_swap_two_adjacent_letters():
    assert 'łsowo' in text_corrector._TextCorrectorPL__swap_two_adjacent_letters('słowo')

def test_qwerty_keyboard_typos():
    assert 'słiwo' in text_corrector._TextCorrectorPL__qwerty_keyboard_typos('słowo')
    assert 'ananaa' in text_corrector._TextCorrectorPL__qwerty_keyboard_typos('ananas')

def test_reduce_qwerty_keyboard_typo():
    assert 'rower' in text_corrector._TextCorrectorPL__reduce_qwerty_keyboard_typo('reower')

def test_wildcards():
    assert '?' in text_corrector._TextCorrectorPL__additional_wildcard('text')[0]
    assert '?' in text_corrector._TextCorrectorPL__replace_letter_with_wildcard('text')[0]
    
def test_check_if_token_is_upper():
    assert text_corrector._TextCorrectorPL__check_if_token_is_upper('low') == False
    assert text_corrector._TextCorrectorPL__check_if_token_is_upper('Big') == True
    assert text_corrector._TextCorrectorPL__check_if_token_is_upper('mEDIUM') == False

def test_split_sequence():
    assert len(text_corrector._TextCorrectorPL__split_sequence('to jest 11nasto elementowy tekst')) == 11
    assert text_corrector._TextCorrectorPL__split_sequence('Testowy, tekst.!//')[-1] == '/'

def test_correct_text():
    text = 'To kest neidbale napisany tekst dla trestu.'
    assert isinstance(text_corrector.correct_text(text), str)
    assert text_corrector.correct_text(text) == 'To jest niedbale napisany tekst dla testu.'
    assert isinstance(text_corrector.correct_text('131424?#$?%@?!'), str)
