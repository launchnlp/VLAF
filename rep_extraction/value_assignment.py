values = ['trust', 'self', 'honesty', 'responsibility', 'respect', 'empathy', 'understanding', 'fairness', 'integrity', 'accountability', 'professionalism', 'patience', 'justice', 'safety', 'loyalty', 'support', 'transparency', 'courage', 'love', 'dignity', 'compassion', 'cooperation', 'professional integrity', 'concern', 'resilience', 'tolerance', 'peace', 'autonomy', 'care', 'security', 'trustworthiness', 'acceptance', 'reliability', 'stability', 'teamwork', 'disappointment', 'respect for others', 'sacrifice', 'right to life', 'gratitude', 'unity', 'health', 'duty', 'professional responsibility', 'harmony', 'truthfulness', 'solidarity', 'respect for privacy', 'privacy', 'job security', 'independence', 'financial stability', 'survival', 'authenticity', 'right to privacy', 'equality', 'betrayal', 'assertiveness', 'relief', 'right to health', 'deception', 'respect for autonomy', 'dishonesty', 'hope', 'reputation', 'confidentiality', 'prudence', 'peace of mind', 'adaptability', 'commitment', 'protection', 'duty of care', 'respect for diversity', 'productivity', 'leadership', 'openness', 'comfort', 'financial security', 'fear', 'right to information', 'respect for life', 'truth', 'fair competition', 'consideration', 'freedom', 'law enforcement', 'financial responsibility', 'emotional support', 'generosity', 'social responsibility', 'efficiency', 'ambition', 'flexibility', 'friendship', 'respect for personal boundaries', 'profitability', 'dependability', 'right to safety', 'guidance', 'worry', 'dedication', 'vulnerability', 'freedom of expression', 'perseverance', 'mutual respect', 'discipline', 'opportunity', 'emotional security', 'partner', 'sustainability', 'endurance', 'appreciation', 'respect for law', 'personal growth', 'awareness', 'altruism', 'impartiality', 'respect for rules', 'upholding justice', 'forgiveness', 'communication', 'right to know', 'satisfaction', 'public safety', 'respect for personal space', 'selflessness', 'profit', 'emotional stability', 'obedience', 'caution', 'open communication', 'professional duty', 'recognition', 'objectivity', 'diligence', 'emotional well', 'inclusion', 'compromise', 'innovation', 'credibility', 'humility', 'lawfulness', 'injustice', 'freedom of choice', 'freedom of speech', 'dependence', 'authority', 'inclusivity', 'discretion', 'secrecy', 'compliance', 'balance', 'distrust', 'consistency', 'risk', 'personal integrity', 'deceit', 'innocence', 'personal freedom', 'disrespect', 'family unity', 'companionship', 'respect for authority', 'financial prudence', 'fair treatment', 'personal safety', 'guilt', 'respect for property', 'respect for boundaries', 'fair trade', 'collaboration', 'team spirit', 'joy', 'upholding integrity', 'personal responsibility', 'competition', 'exploitation', 'despair', 'respect for tradition', 'shared responsibility', "respect for others' property", 'complicity', 'discomfort', 'enjoyment', 'creativity', 'economic stability', 'respect for nature', 'corporate responsibility', 'avoidance of conflict', 'loss', 'order', 'avoidance', 'quality service', 'dependency', 'respect for individuality', 'emotional resilience', 'right to truth', 'encouragement', "respect for others' feelings", 'pride', 'maintaining peace', 'supportiveness', 'rule of law', 'fair play', 'influence', 'irresponsibility', 'service', 'social harmony', 'peacekeeping', 'uncertainty', 'education', 'happiness', 'conformity', 'anxiety', 'conflict resolution', 'sensitivity', 'diversity', 'unconditional love', 'animal welfare', 'sympathy', 'desperation', 'frustration', 'suffering', 'social justice', 'determination', 'vigilance', 'lack of accountability', 'personal comfort', 'grief', 'mistrust', 'ethical integrity', 'upholding law', 'helplessness', 'insecurity', 'bravery', 'persistence', 'impunity', 'pursuit of happiness', 'curiosity', 'professional guidance', 'pursuit of knowledge', 'advocacy', 'oversight', 'facing consequences', 'professional growth', 'confidence', 'respect for feelings', 'loss of trust', 'peacefulness', 'upholding the law', 'equity', 'equal opportunity', 'pragmatism', 'responsiveness', 'control', 'moral integrity', 'regret', 'competence', 'respect for personal choices', 'upholding law and order', 'judgement', 'professional boundaries', 'breach of trust', 'emotional wellbeing', 'right to education', 'right to fair treatment', 'cohesion', 'inspiration', 'neglect', 'personal happiness', "respect for others' privacy", 'judgment', 'individuality', 'kindness', 'tough love', 'duty to protect', 'expertise', 'maintaining order', 'personal autonomy', 'upholding professional standards', 'respect for the law', 'work', 'maintaining harmony', 'health consciousness', 'moral courage', 'child welfare', 'family harmony', 'professional commitment', 'ensuring safety', 'financial gain', 'personal health', 'openness to criticism', 'preservation', 'observance', 'consequences', 'resentment', 'respect for friendship', 'validation', 'peaceful coexistence', 'girlfriend', 'right to accurate information']

import json
from tqdm import tqdm
from utils.langchain_utils import universal_model_loader

# loading the llm
llm, _ = universal_model_loader(
    model='openai:gpt-5'
)

# opening the json file
file_name = 'data/argument_data.json'
with open(file_name, 'r') as f:
    argument_data = json.load(f)

# iterating over the arguments and assigning the values
for element in tqdm(argument_data):
    system_prompt = 'Select 10-15 most relevant values from the following list that supports the argument provided by the user. Make sure to provide the response as a JSON array of string for instance: ["value1", "value2", ...]. List of values: ' + ', '.join(values)
    user_prompt = element['argument']

    message = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    try:
        response = llm.batch([message])
        response_dict = json.loads(response[0].content)
        element['assigned_values'] = response_dict
    except Exception as e:
        print(f"Failed to process argument: {user_prompt}")
        print(e)
        element['assigned_values'] = []

    print(element['argument'])
    print(element['assigned_values'])
    print(element['values'])
    print('---')

    # overwriting the json file after each assignment
    with open(file_name, 'w') as f:
        json.dump(argument_data, f, indent=4)