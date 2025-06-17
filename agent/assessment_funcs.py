import logging

logger = logging.getLogger(__name__)


def score_wrapper(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return {"reason": "", "answer": result}

    return wrapper


@score_wrapper
def CAPS5_rule(var_template, prerequisites):
    intensity, num = prerequisites.values()
    if intensity is None:
        logger.warning(f"Missing intensity for {var_template}")
        intensity = 0
    if num is None:
        logger.warning(f"Missing num for {var_template}")
        num = 0
    keywords = var_template.patterns["keywords"]
    if keywords == "frequency":
        if intensity == 0:
            score = 0
        elif intensity == 1:
            score = 1 if num > 0 else 0
        elif intensity == 2:
            score = 2 if num >= 2 else 1
        elif intensity == 3:
            score = 3 if num >= 8 else 2
        elif intensity == 4:
            score = 4 if num >= 15 else 3
    elif keywords == "percentage":
        if intensity == 0:
            score = 0
        elif intensity == 1:
            score = 1 if num > 0 else 0
        elif intensity == 2:
            score = 2 if num >= 20 else 1
        elif intensity == 3:
            score = 3 if num >= 50 else 2
        elif intensity == 4:
            score = 4 if num >= 70 else 3
    elif keywords == "number of important parts":
        if intensity == 0:
            score = 0
        elif intensity == 1:
            score = 1 if num > 0 else 0
        elif intensity == 2:
            score = 2 if num >= 0 else 1
        elif intensity == 3:
            score = 3 if num >= 1 else 2
        elif intensity == 4:
            score = 4 if num >= 1 else 3
    else:
        if intensity > 0 or num > 0:
            score = 1
        else:
            score = 0
    return score


@score_wrapper
def dsmcaps_critf_admin(prerequisites):
    scores = [score for vid, score in prerequisites.items() if vid.endswith("distress")]
    return 1 if any([score > 0 for score in scores]) else 0
