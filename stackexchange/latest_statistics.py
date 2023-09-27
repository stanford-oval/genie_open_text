"""
Get the latest StackExchange statistics from https://stackexchange.com/sites.

Download the html file of https://stackexchange.com/sites to HTML_PATH.
Run python latest_statistics.py --html_path HTML_PATH --result_path RESULT_PATH
"""
from argparse import ArgumentParser

import pandas as pd
from bs4 import BeautifulSoup


def main(args):
    file_path = args.html_path

    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'lxml')

    domains_info = []

    # Find all the 'a' tags with class 'noscript-link'
    for link in soup.find_all('a', class_='noscript-link'):
        # Extract the site name and description
        site_name = link.find("div", class_="gv-expanded-site-name").text
        site_description = link.find("div", class_="gv-expanded-site-description").text
        site_url = link['href']

        # Extract statistics
        stats = {}
        table = link.find("table", class_="gv-stats")
        for row in table.find_all("tr"):
            key = row.find("th").text
            value_span = row.find("td").find("span")

            # Extract the exact numbers from the title attribute if available, otherwise use the text
            if value_span and "title" in value_span.attrs:
                value = value_span["title"].split()[0].replace(',', '')  # Remove comma for thousands
            else:
                value = row.find("td").text

            # Store the extracted values
            stats[key] = value

        # Append all the extracted data for each domain to a list
        domain_data = {
            'Site Name': site_name,
            'Site Description': site_description,
            'Site URL': site_url,
            'Number of Questions': stats.get('questions', 'N/A'),
            'Number of Answers': stats.get('answers', 'N/A'),
            'Number of Users': stats.get('users', 'N/A')
        }

        domains_info.append(domain_data)

    print(f"#Total domain: {len(domains_info)}")

    df = pd.DataFrame(domains_info)
    df.to_csv(args.result_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--html_path', type=str)
    parser.add_argument('--result_path', type=str, help='The results will be saved into a csv file.')

    main(parser.parse_args())
