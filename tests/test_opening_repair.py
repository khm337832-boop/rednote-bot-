import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("BOT_TOKEN", "")

import bot


def test_invalid_card_introduction_gets_real_self_introduction():
    name = "အေဒင်"
    context = {
        "protagonist": name,
        "protagonist_pronoun": "ကျွန်တော်",
        "character_cards": [{
            "spoken_name": name,
            "introduction": "အေဒင်က ကျွန်တော့်ကို ကူညီခဲ့တယ်။",
        }],
    }
    scenes, _ = bot._ensure_first_person_protagonist_scenes(
        [{"burmese": "ကျွန်တော် ရွာထဲကို ပြန်လာခဲ့တယ်။"}],
        protagonist=name, story_context=context, protagonist_pronoun="ကျွန်တော်",
        is_opening=True,
    )
    assert bot._protagonist_self_introduction_present(
        scenes, protagonist_forms=[name], expected_pronoun="ကျွန်တော်"
    )
    again, _ = bot._ensure_first_person_protagonist_scenes(
        scenes, protagonist=name, story_context=context,
        protagonist_pronoun="ကျွန်တော်", is_opening=True,
    )
    assert again == scenes


def test_foreign_evidence_is_not_inserted_as_spoken_introduction():
    intro = bot._build_protagonist_self_introduction("董礼茂", "ကျွန်တော်")
    assert intro == ""


def test_quoted_card_introduction_is_not_reused():
    context = {"character_cards": [{
        "spoken_name": "အေဒင်",
        "introduction": "သူက ‘ကျွန်တော့်အမည်က အေဒင် ဖြစ်တယ်’ လို့ ပြောတယ်။",
    }]}
    intro = bot._build_protagonist_self_introduction(
        "အေဒင်", "ကျွန်တော်", story_context=context
    )
    assert bot._protagonist_self_introduction_present(
        [intro], protagonist_forms=["အေဒင်"], expected_pronoun="ကျွန်တော်"
    )
