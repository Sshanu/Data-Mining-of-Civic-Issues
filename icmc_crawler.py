import urllib.request, json 
import requests
from elasticsearch import Elasticsearch
import time
from utils_icmc import set_category_field, set_time_field


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 

url = "http://api.ichangemycity.com/api/complaints/list?page="
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

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
		res = (es.search(index="data_icmc", doc_type = "icmc", body=query))
		return res['hits']['total']
	else:
		print("local host not running")


while(1):
	date_today = time.strftime("%Y-%m-%d")
	filename_1 = "data/icmc/complaints_" + date_today + ".json"
	error_filename = "data/icmc/error_complaints_" + date_today + ".json"
	error_f = open(error_filename, "w")
	f_1 = open(filename_1,"w")
	next_day = 0
	qdict = {}
	count = 0
	dup_count = 0
	pg = 0

	while(1):
		pg = pg + 1
		print(pg)
		k = 0
		while k<1:
			if(k<=-5):
				error_f.write(str(pg))
				error_f.write("\n")
				pg += 1
				print(pg)
				k = 0
			try:
				cur_url = (url + str(pg)) 
				request=urllib.request.Request(cur_url,None,headers)
				response = urllib.request.urlopen(request)
				entire_data = response.read()
				k = 1
			except:
				print("Internet Down!! Reconnect ASAP")
				k -= 1
				time.sleep(60)
				continue
		try:
			json_data = json.loads(entire_data.decode())
			count = 0
		except:
			print("Error in load")
			print(entire_data)
			print()
			time.sleep(10)
			count = count + 1
			if count<2:
				pg = pg - 1
			else:
				count = 0
			continue

		try:
			for complaint in json_data['data']:
				if complaint['postedOn'].split(' ')[1][0:3] == "day":
					if str(check_duplicate(str(complaint['id'])))!="0":
						dup_count = dup_count + 1
					else:
						dup_count = 0
				if dup_count>3:
					next_day = 1
					break
				qdict = {}
				qdict['id'] = complaint['id']
				qdict['description'] = complaint['description']
				qdict['location'] = complaint['location']['name']
				qdict['image_url'] = []
				for l, pic in enumerate(complaint['pictures']):
					image_url = pic['url']
					qdict['image_url'].append(image_url)
					urllib.request.urlretrieve(image_url, "data/icmc/images/" + str(complaint['id']) + "_" + str(l) + ".jpg")
				qdict['postedOn'] = complaint['postedOn']
				qdict['category'] = complaint['category']
				qdict['subCategory'] = complaint['subCategory']
				qdict = set_time_field(qdict)
				qdict = set_category_field(qdict)
				qdict['page'] = pg
				es.index(index='data_icmc', doc_type='icmc', body=qdict)
				json.dump(qdict,f_1)
				f_1.write("\n")
		except:
			print("Error in For loop")
			continue
		if next_day==1:
			break
	f_1.close()
	filename = "data/icmc/done_complaints__"+ date_today +".txt"
	f = open(filename,"w")
	f.write("Done")
	f.close()
	while(date_today == time.strftime("%Y-%m-%d")):
		time.sleep(3600)