import os
with open('API_KEYS', 'r') as key_file:
    for line in key_file:
        key, val = line.split()[-1].split('=', maxsplit=1)
        os.environ[key] = val

from llm import llm_generate, get_total_cost
from utils import extract_year



def test_extract_date():
    passages = [
        "Phoenix Mills [SEP] Phoenix Mills Ltd was incorporated in the year 1905.",
        "Phoenix Marketcity [SEP] Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.",
        "Starbucks [SEP] As of November 2021, the company had 33,833 stores in 80 countries, 15,444 of which were located in the United States.",
        "1984 eruption of Mauna Loa [SEP] The 1984 eruption of Mauna Loa was a Hawaiian eruption in the U.S. state of Hawaii that lasted from March 25 to April 15, 1984. It ended a 9-year period of quiescence at the volcano and continued for 22 days, during which time lava flows and lava fountains issued from the summit caldera and fissures along the northeast and southwest rift zones. Although the lava threatened Hilo, the flow stopped before reaching the outskirts of town.",
        "Chinese government response to COVID-19 [SEP] 2020 Chinese New Year. The Wuhan government, which announced a number of new measures such as cancelling the Chinese New Year celebrations, in addition to measures such as checking the temperature of passengers at transport terminals first introduced on 14 January.  The leading group decided to extend the Spring Festival holiday to contain the outbreak.",
        "On November 29, 2021, Jack Dorsey stepped down as CEO. He was replaced by CTO Parag Agrawal. On October 27, 2022, Elon Musk closed a deal to purchase the company and fired Agrawal, CFO Ned Segal, chief legal officer Vijaya Gadde, and general counsel Sean Edgett. Musk replaced the prior board as the sole director of Twitter and appointed himself as the CEO. Finances. Funding. Twitter raised over US$57\u00a0million from venture capitalist growth funding, although exact figures are not publicly disclosed. Twitter's first A round of funding was for an undisclosed amount that is rumored to have been between US$1\u00a0million and US$5\u00a0million.",
        "Chinese New Year was on February 5th in 2023.",
        "The December 2021 tornado outbreak was devastating for the areas that were affected.",
        "It was conducted at Stanford University in 1971, by a research group led by psychology professor Philip Zimbardo using college students. In the study, volunteers were randomly assigned to be either guards or prisoners in a mock prison, with Zimbardo himself serving as the superintendent.",
        "Serena Williams announced her impending evolution away from professional tennis in 2022.",
        "List of accolades received by Hayao Miyazaki [SEP] is a Japanese film director, producer, screenwriter, animator, author, and manga artist. A co-founder of Studio Ghibli, a film and animation studio, he has attained international acclaim as a masterful storyteller and as a maker of anime feature films. His works are characterized by the recurrence of progressive themes, such as environmentalism, pacifism, feminism, love and family. His films' protagonists are often strong girls or young women, and several of his films present morally ambiguous antagonists with redeeming qualities. In the course of his career, Miyazaki has received multiple awards and nominations",
        "The Wind Rises was released in Japan on 20 July 2013. The film played in competition at the 70th Venice International Film Festival and had its official North American premiere at the 2013 Toronto International Film Festival. The Japanese DVD release sold 128,784 units until 7 December 2014. In the United States, Walt Disney Studios Home Entertainment released the film on Blu-ray Disc and DVD on 18 November 2014.",
        "Haus Vaterland was a pleasure palace in central Berlin that was famous for its large size and variety of attractions. It was partially destroyed in World War II, but reopened in a limited form until 1953. It was finally demolished in 1963.",
        "Cousins was released from prison in January 2018.[105] As part of his parole conditions, Cousins began part-time work in the West Coast Eagles' community and game development department, but abruptly left the job in April 2018. In August 2018, he was arrested and charged with multiple offences, including drug possession, making threats and breaking a violence restraining order.[107] He was released on bail in April 2019.[108]",
        "Miyazaki's later films—Howl's Moving Castle (2004), Ponyo (2008), and The Wind Rises (2013)—also enjoyed critical and commercial success. Following the release of The Wind Rises, Miyazaki announced his retirement from feature films, though he returned in 2016 to work on the upcoming feature film How Do You Live? (2023).",
        "Significant events included Elizabeth's coronation in 1953 and the celebrations of her Silver, Golden, Diamond, and Platinum jubilees in 1977, 2002, 2012, and 2022, respectively. Although she faced occasional republican sentiment and media criticism of her family—particularly after the breakdowns of her children's marriages, her annus horribilis in 1992, and the death in 1997 of her former daughter-in-law Diana—support for the monarchy in the United Kingdom remained consistently high throughout her lifetime, as did her personal popularity. Elizabeth died at Balmoral Castle, Aberdeenshire, in 2022 at the age of 96 and was succeeded by her eldest son, Charles III.",
        "I enjoy watching the 2006–07 season.",
        "she is the prime minister from 1907 to 1937.",
        "Earthquake in Japan On 16 August 2005, an earthquake struck the east coast of the Japanese island of Honshū at 11.46 am (02:46 UTC) on 16 August 2005, causing damage and power outages. The event registered 7.2 on the moment magnitude scale. Earthquake characteristics. The earthquake occurred on Tuesday, August 16, 2005, and affected Japan's northeastern coast. It triggered a tsunami warning, and buildings shook 200 miles away in the capital, Tokyo. It was initially estimated to have a moment magnitude of 6.8, and the United States Geological Survey later declared it as a 7.2. Damage and casualties.",
        "Earthquake in Iran The 2005 Qeshm earthquake occurred on November 27 at 13:52 IRST (10:22 UTC) on the sparsely populated Qeshm Island off Southern Iran, killing 13 people and devastating 13 villages. It was Iran's second major earthquake of 2005, following the one at Zarand in February. The epicenter was about south of Tehran, close to Iran's southern borders. Initial measurements showed that the earthquake registered about 6.0 on the moment magnitude scale, although that was reduced to 5.8 after further analysis. More than 400 minor aftershocks followed the main quake, 36 of which were greater than magnitude 2.5. The earthquake occurred in a remote area during the middle of the day, limiting the number of fatalities.",
    ]
    
    for passage in passages:
        print(extract_year(title='', passage=passage))

def test_ban_line_break_start():
    llm_input1 = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Assistant, and a human user, called User.
In the following interactions, User and Assitant will converse in natural language, and Assistant will do its best to answer User's questions.
Assistant knows a lot, and always tells the truth.
Assistant can use Google to get external knowledge if needed. It can use Search Query to issue search queries and get responses from Google. Assistant carefully anwers based on the search results, providing only information supported by the results.
The current year is 2023, and in particular today's date is 2023-03-06. Assistant's responses reflect this.

=====
User: What new discoveries from James Webb Space telescope can I tell my 9 year old about?
Assistant: Your 9-year old would be fascinated to hear that the James Webb Space Telescope has discovered farthest galaxies ever found. It has found galaxies so far away that it took 13.4 billion years for their lights to reach the telescope. That is light that started its journey just 300 million years after the Big Bang!
User: what is big bang?
Assistant: The Big Bang is the prevailing cosmological model for the universe from the earliest known periods through its subsequent large-scale evolution. In layman's terms, it is the explosion that created our universe.
[You Google "new discoveries from James Webb Space telescope for kids" to get more information]
[You get this information:
The James Webb Space Telescope has detected galaxies more than 13.4 billion light years away, less than 300 million years after the Big Bang. This is a new record and astronomers are excited to learn more about these distant galaxies. The telescope is named after James E. Webb, who was the administrator of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.]
User: what else can I tell my child about?
Assistant: The James Webb Space telescope is named after James E. Webb, who was the administrator of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.
=====
User: did you watch the australian open 2023?
Assistant: yeah, it was exciting.
[You Google "australian open 2023 champion" to get more information]
[You get this information:
Novak Djokovic won the men's singles tennis title at the 2021 Australian Open. It was his record-extending ninth Australian Open title and his 18th major title overall. With his fourth round win against Milos Raonic, Djokovic joined Roger Federer as only the second man with 300 or more match wins in major singles competitions.]
User: Who won?
Assistant: I'm not sure. I only know that Djokovic won in 2021, but I'm not so sure about 2023.
=====
User: So what do you think about Elon Musk buying Twitter?
Assistant: I think it's a good idea.
User: Why?
Assistant: There are a few reasons why I think it's a good idea.
User: One reason is that it would give him more control over his public image.
Assistant:"""

    llm_input2 = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Assistant, and a human user, called User.
In the following interactions, User and Assitant will converse in natural language, and Assistant will do its best to answer User's questions.
Assistant knows a lot, and always tells the truth.
Assistant can use Google to get external knowledge if needed. It can use Search Query to issue search queries and get responses from Google. Assistant carefully anwers based on the search results, providing only information supported by the results.
The current year is 2023, and in particular today's date is 2023-03-06. Assistant's responses reflect this.

=====
User: What new discoveries from James Webb Space telescope can I tell my 9 year old about?
Assistant: Your 9-year old would be fascinated to hear that the James Webb Space Telescope has discovered farthest galaxies ever found. It has found galaxies so far away that it took 13.4 billion years for their lights to reach the telescope. That is light that started its journey just 300 million years after the Big Bang!
User: what is big bang?
Assistant: The Big Bang is the prevailing cosmological model for the universe from the earliest known periods through its subsequent large-scale evolution. In layman's terms, it is the explosion that created our universe.
[You Google "new discoveries from James Webb Space telescope for kids" to get more information]
[You get this information:
The James Webb Space Telescope has detected galaxies more than 13.4 billion light years away, less than 300 million years after the Big Bang. This is a new record and astronomers are excited to learn more about these distant galaxies. The telescope is named after James E. Webb, who was the administrator of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.]
User: what else can I tell my child about?
Assistant: The James Webb Space telescope is named after James E. Webb, who was the administrator of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.
=====
User: did you watch the australian open 2023?
Assistant: yeah, it was exciting.
[You Google "australian open 2023 champion" to get more information]
[You get this information:
Novak Djokovic won the men's singles tennis title at the 2021 Australian Open. It was his record-extending ninth Australian Open title and his 18th major title overall. With his fourth round win against Milos Raonic, Djokovic joined Roger Federer as only the second man with 300 or more match wins in major singles competitions.]
User: Who won?
Assistant:"""

    reply = llm_generate(template_file=None, prompt_parameter_values=None, filled_prompt=llm_input1,
                    engine='text-davinci-002',
                    max_tokens=120, temperature=0.0, stop_tokens=None,
                    top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0,
                    postprocess=False, ban_line_break_start=False)

    print('ban_line_break_start=False:' + reply)

    reply = llm_generate(template_file=None, prompt_parameter_values=None, filled_prompt=llm_input1,
                engine='text-davinci-002',
                max_tokens=120, temperature=0.0, stop_tokens=None,
                top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0,
                postprocess=False, ban_line_break_start=True)

    print('ban_line_break_start=True:' + reply)


    reply = llm_generate(template_file=None, prompt_parameter_values=None, filled_prompt=[llm_input1, llm_input2],
            engine='text-davinci-002',
            max_tokens=120, temperature=0.0, stop_tokens=None,
            top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0,
            postprocess=False, ban_line_break_start=True)

    print('ban_line_break_start=True, list input:' + str(reply))

def test_llm_caching():
    llm_input = "Hi, my name is"
    print('Now testing with temperature = 0')
    for _ in range(5):
        reply = llm_generate(template_file=None, prompt_parameter_values=None, filled_prompt=llm_input, stop_tokens=None, max_tokens=10,
            engine='text-davinci-002', temperature=0)
        print(reply)
    print('total_cost = ', get_total_cost())
    print('='*10)
    print('Now testing with temperature > 0')
    for _ in range(5):
        reply = llm_generate(template_file=None, prompt_parameter_values=None, filled_prompt=llm_input, stop_tokens=None, max_tokens=10,
            engine='text-davinci-002', temperature=1)
        print(reply)
    print('total_cost = ', get_total_cost())

if __name__ == "__main__":
    # test_ban_line_break_start()
    # test_extract_date()
    test_llm_caching()