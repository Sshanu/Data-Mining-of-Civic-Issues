from datetime import date, timedelta, datetime
import time
import requests
import json

def convert(s):
	current_time = datetime.now()
	new_time = current_time
	s = s.split(' ')
	if s[1][0:3]=="day":
		new_time = current_time - timedelta(days=int(s[0]))
	elif s[1][0:4]=="week":
		new_time = current_time - timedelta(weeks=int(s[0]))
	elif s[1][0:4]=="hour":
		new_time = current_time - timedelta(hours=int(s[0]))
	elif s[1][0:4]=="year":
		new_time = current_time - timedelta(days=365*int(s[0]))
	elif s[1][0:5]=="month":
		new_time = current_time - timedelta(days=30*int(s[0]))
	elif s[1][0:6]=="second":
		new_time = current_time - timedelta(seconds=int(s[0]))
	elif s[1][0:6]=="minute":
		new_time = current_time - timedelta(minutes=int(s[0]))
	return str(new_time)

def set_time_field(qdict):
	qdict['postedOn'] = convert(qdict['postedOn']).replace(" "," T")[:-7]
	return qdict

def set_category_field(new_json):
	category = new_json['subCategory']
	flag = 0
	if category=="Garbage" or category=="Garbage and Unsanitary Practices - Others":
		new_json["myCategory"] = "Garbage"
		flag = 1
	if category=="Bad Roads" or category=="Maintenance of Roads and Footpaths - Others" or category=="Potholes" or category=="Footpaths":
		new_json["myCategory"] = "Bad Roads and Footpath"
		flag = 1
	if category=="Need New Streetlights" or category=="Repair of streetlights":
		new_json["myCategory"] = "Streetlights"
		flag = 1
	if category=="Maintenance of Lakes" or category=="Lakes - Others":
		new_json["myCategory"] = "Lakes"
		flag = 1
	if category=="Trees  Parks and Playgrounds - Others" or category=="Parks and playgrounds":
		new_json["myCategory"] = "Trees, Parks and Playgrounds"
		flag = 1
	if category=="Overflow of Storm Water Drains" or category=="Flooding of Roads and Footpaths" or category=="No Sewage Drains" or category=="Sewage and Storm Water Drains - Others":
		new_json["myCategory"] = "Sewage Drains"
		flag = 1
	if category=="Water Supply":
		new_json["myCategory"] = "Water Supply"
		flag = 1
	if category=="Water Leakage":
		new_json["myCategory"] = "Water Leakage"
		flag = 1
	if category=="Electricity and Power Supply – Others" or category=="Electricity":
		new_json["myCategory"] = "Electricity"
		flag = 1
	if category=="Illegal posters and Hoardings" or category=="Hoardings":
		new_json["myCategory"] = "Hoardings"
		flag = 1
	if category=="Air Pollution":
		new_json["myCategory"] = "Air Pollution"
		flag = 1
	if category=="Noise Pollution":
		new_json["myCategory"] = "Noise Pollution"
		flag = 1
	if category=="Mosquitos":
		new_json["myCategory"] = "Mosquitos"
		flag = 1
	if category=="Stray Dogs":
		new_json["myCategory"] = "Stray Dogs"
		flag = 1
	if flag==0:
		new_json["myCategory"] = new_json["subCategory"]
	return new_json
