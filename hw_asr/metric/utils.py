# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.0
    edit_dist = editdistance.eval(target_text, predicted_text)
    return edit_dist / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.0
    target_char = target_text.split()
    pred_char = predicted_text.split()
    edit_dist = editdistance.eval(target_char, pred_char)
    return edit_dist / len(target_text.split())
