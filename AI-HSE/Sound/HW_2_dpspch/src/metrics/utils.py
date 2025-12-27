import editdistance
# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    # TODO
    if target_text == '':
        if predicted_text == '':
            return 0
        else:
            return 1

    return editdistance.eval(predicted_text, target_text) / len(target_text)
    

def calc_wer(target_text, predicted_text) -> float:
    # TODO
    if target_text == '':
        if predicted_text == '':
            return 0
        else:
            return 1
        
    return editdistance.eval(predicted_text.split(), target_text.split()) / len(target_text.split())
