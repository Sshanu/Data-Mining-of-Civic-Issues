import urllib.request, json 
import requests
from elasticsearch import Elasticsearch
import time
import re
from urllib.request import urlopen as uReq
import calendar
import datetime
import sys
from datetime import date, timedelta, datetime
from bs4 import BeautifulSoup as soup
import pickle
from tqdm import tqdm

sys.setrecursionlimit(1189285)
month_dict = dict((v,k) for k,v in enumerate(calendar.month_name))

def html_extractor(site_url):
	k = 0
	while k<1:
		try:
			uClient = uReq(site_url)
			page_html = uClient.read()
			uClient.close()
			k = k+1
		except:
			print("Internet Down!! Reconnect ASAP", site_url)
			k = -1
			time.sleep(60)
			continue
	return page_html

def info_extractor(complaint, page):
	problem_div = page.find("div", {"class": "problem-header clearfix"})
	p = re.compile('([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]')
	problem_meta_info = problem_div.find("p", {"class": "report_meta_info"}).text
	time_check = p.search(problem_meta_info)
	time_end = time_check.span()[1]
	time_val = time_check.group()
	datetime_str = problem_meta_info[time_end+1:]
	try:
		today_check = re.search("today", datetime_str)
		complaint['postedOn'] = str(datetime.date.today()) + "T" + time_val
	except:
		year_check = re.search("\s\d\d\d\d\s", datetime_str)
		datetime_str = datetime_str[:year_check.span()[1]-1]
		day_check = re.search("\s\d?\d\s", datetime_str)
		day_val = day_check.group()[1:-1].zfill(2)
		day_end = day_check.span()[1]
		month_str = datetime_str[day_end:-5]
		month_val = str(month_dict[month_str]).zfill(2)
		year_val = datetime_str[-4:]
		complaint["postedOn"] = year_val + "-" + month_val + "-" + day_val + "T" + time_val
	complaint["title"] = problem_div.find("h1", {"class": "moderate-display"}).text
	category_check = re.search("(?<=in the )(\S|\s)+(?= category)", problem_meta_info)
	try:
		complaint["category"] = category_check.group()
	except:
		complaint["category"] = None
	try:
		complaint["images"] = [p.find("a").get("href") for p in problem_div.find_all("div", {"class": "update-img"})]
		for l, img_url in enumerate(complaint['images']):
			urllib.request.urlretrieve("https://www.fixmystreet.com/" + img_url, "data/fms/images/" + str(complaint['id']) + "_" + str(l) + ".jpg")
	except: 
		complaint["images"] = None
	complaint["descroption"] = [p.string for p in problem_div.find("div", {"class": "moderate-display"}).find_all("p")]
	return complaint

def check_duplicate(new_id):
	r = requests.get('http://localhost:9200') 
	if r.status_code == 200:
		query = {
					"query":{
						"match" : {
							"id" : new_id
						}
					},
					"size":5,
					"_source": ["postedOn"]
				}
		try:
			res = (es.search(index="data", doc_type = "fms", body=query))
			return res['hits']['total']
		except:
			return 0
	else:
		print("local host not running")

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
cond = "https://www.fixmystreet.com/report"        
url = "https://www.fixmystreet.com/reports/"

while(1):
	date_today = time.strftime("%Y-%m-%d")
	filename_1 = "data/fms/complaints_"+date_today+".json"
	f_1 = open(filename_1,"w")
	error_filename = "data/fms/error_complaints__"+ date_today +".txt"
	error_f = open(error_filename, "w")

	page_html = html_extractor(url)
	page_soup = soup(page_html,"html.parser")
	cities = []
	for att in page_soup.find_all('option'):
		if(att.get('value')!=""):
			cities.append(att.string)
	print("Total no. of cities: ", len(cities))

	for i, city in enumerate(tqdm(cities)):
		reports_links = []
		pg = 0
		dup_count = 0
		next_day = 0

		while(1):
			pg +=1
			print("Current city: ", city, "Page no: ", pg)
			city_page_html = html_extractor(url+urllib.parse.quote_plus(city)+"?p="+str(pg))
			city_page_soup = soup(city_page_html,"html.parser")
			pins = city_page_soup.find("div", {"id": "pins"})
			if(pins==None):
				break
			reports_tags = pins.find_all("a")
			if(reports_tags==[]):
				break
			for reports in tqdm(reports_tags):
				href_val = reports.get("href")
				if(href_val.find(cond)==0):
					if(href_val not in reports_links):
						reports_links.append([href_val, city])
						report_page_html = html_extractor(href_val)
						report_page_soup = soup(report_page_html,"html.parser")
						problem_div = report_page_soup.find("div", {"class": "problem-header clearfix"})
						update_div = report_page_soup.find("ul", {"class": "item-list item-list--updates"})
						complaint = {}
						complaint["location"] = city
						complaint["url"] = href_val
						complaint["id"] = href_val[35:]
						try:
							complaint = info_extractor(complaint, report_page_soup)
							error_flag = 0
						except:
							error_flag = 1
							error_f.write(href_val)
							error_f.write("\n")	
						if str(check_duplicate(str(complaint['id'])))!="0":
							dup_count = dup_count + 1
						else:
							dup_count = 0
							if(error_flag==0):
								es.index(index='data', doc_type='fms', body=complaint)
								json.dump(complaint, f_1)
								f_1.write("\n")
						if dup_count>3:
							next_day = 1
							break	
			if next_day==1:
					break

	f_1.close()
	error_f.close()
	filename = "data/fms/done_complaints__"+ date_today +".txt"
	f = open(filename,"w")
	f.write("Done")
	f.close()
	while(date_today == time.strftime("%Y-%m-%d")):
		time.sleep(3600)