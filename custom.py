import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForQuestionAnswering
import torch


url = 'https://brainlox.com/courses/category/technical'

response = requests.get(url)


if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    course_elements = soup.find_all('div', class_='course')  

    course_data = []

    for element in course_elements:
        title = element.find('h2').text.strip()

        description = element.find('p', class_='course-description').text.strip() 

        link = element.find('a')['href']

        course_data.append({
            'title': title,
            'description': description,
            'link': link
        })


    for course in course_data:
        print(f'Title: {course["title"]}')
        print(f'Description: {course["description"]}')
        print(f'Link: {course["link"]}')
        print('-' * 50)

else:
    print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

def process_user_input(user_input, course_data):
    tokens = tokenizer.encode("Find courses about " + user_input, return_tensors="pt")

    answer_start_scores, answer_end_scores = model(**tokens)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens[0][answer_start:answer_end]))

    recommended_courses = [course for course in course_data if answer.lower() in course['title'].lower() or answer.lower() in course['description'].lower()]

    return recommended_courses

def recommend_courses(user_query, course_data):
    recommended_courses = []
    
    user_query = user_query.lower()
    
    for course in course_data:
        if user_query in course['title'].lower() or user_query in course['description'].lower():
            recommended_courses.append(course)
    
    return recommended_courses

while True:
    user_input = input("Ask the chatbot for a technical course recommendation (type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    recommended_courses = recommend_courses(user_input, course_data)
    
    if recommended_courses:
        print("Recommended Courses:")
        for course in recommended_courses:
            print(f"Title: {course['title']}")
            print(f"Description: {course['description']}")
            print(f"Link: {course['link']}")
            print('-' * 50)
    else:
        print("No courses found matching your query.")
