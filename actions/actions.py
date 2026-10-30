"""
Rasa Custom Action Server
=========================
This module contains the custom actions for a Task-Oriented Conversational System.
It acts as the bridge between the Rasa dialogue management engine and external APIs.

Key functionalities include:
- Weather API Integration: Fetches geopolitical weather forecasts.
- Calendar API Integration: Executes CRUD (Create, Read, Update, Delete) operations.
- Robust Form Validation: Implements "Time-Travel" logic to gracefully handle user 
  interruptions, context-switching, and slot-overwrites during multi-turn scheduling.
- Natural Language Date Parsing: Utilizes `dateparser` and custom logic to translate
  conversational dates (e.g., "next tuesday") into structured API formats.

Security Note: 
Live API endpoints have been abstracted using environment variables for public deployment.
To run this locally, define these variables in your environment or a `.env` file.
"""

from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet, ActiveLoop, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
import requests
import dateparser
import datetime
import logging
import re
import calendar
import os

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Secured via environment variables to protect private endpoints
CALENDAR_ID = os.getenv("CALENDAR_ID", "DEFAULT_USER_ID")
CALENDAR_API_URL = os.getenv("CALENDAR_API_URL", "https://api.example.com/calendar")
WEATHER_API_URL = os.getenv("WEATHER_API_URL", "https://api.example.com/weather")

# --- HELPER Function: Check if text is a date ---
def looks_like_date(text: Any) -> bool:
    if not text:
        return False
    
    text_lower = str(text).lower().strip()

    date_keywords = ['today', 'tomorrow', 'yesterday',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'next', 'last', 'this', 'week', 'month', 'day',
        'am', 'pm', 'morning', 'afternoon', 'evening', 'night'
    ]

    date_word = any(keyword in text_lower for keyword in date_keywords)
    numeric_word = any(char.isdigit() for char in text_lower)
    
    parsed = dateparser.parse(text_lower, settings= {'PREFER_DATES_FROM':'future'})
    return bool(parsed and (date_word or numeric_word))

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def valid_day_in_month(user_text: str, parsed: datetime.datetime) -> bool:
    t = user_text.lower()

    # detect mentioned month
    mentioned_month = None
    for m, num in MONTHS.items():
        if m in t:
            mentioned_month = num
            break

    # detect mentioned day number (e.g., 14th, 14)
    m_day = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\b", t)
    mentioned_day = int(m_day.group(1)) if m_day else None

    if mentioned_month is not None:
        if parsed.month != mentioned_month:
            return False

        if mentioned_day is not None:
            max_day = calendar.monthrange(parsed.year, mentioned_month)[1]
            if mentioned_day < 1 or mentioned_day > max_day:
                return False

    return True

# --- HELPER Function: Date Parser ---
def parse_date(date_text: Any) -> Optional[datetime.datetime]:
    """Parse date from either Duckling (ISO) or natural language"""
    if not date_text:
        return None
    
    text_str = str(date_text).strip()
    logger.debug(f"parse_date - Input: {text_str}, Type: {type(date_text)}")

    # 1. Try parsing ISO format from Duckling
    try:
        if "T" in text_str:
            # Remove timezone suffix
            iso_str = text_str.split("+")[0].split("Z")[0]
            parsed = datetime.datetime.fromisoformat(iso_str)
            logger.debug(f"parse_date - Parsed from ISO: {parsed}")
            return parsed
    except Exception as e:
        logger.debug(f"parse_date - ISO parsing failed: {e}")
        pass
    
    # 2. Fallback to dateparser
    parsed = dateparser.parse(
        text_str,
        settings={'PREFER_DATES_FROM': 'future'}
    )
    
    logger.debug(f"parse_date - Parsed from dateparser: {parsed}")
    return parsed

# --- HELPER Function: Friendly Date Conversion ---
def format_friendly_date(date_obj):
    """Converts a date to 'Today', 'Tomorrow' or 'Monday'"""
    today = datetime.datetime.now().date()
    target = date_obj.date()
    if target == today:
        return "today"
    elif target == today + datetime.timedelta(days=1):
        return "tomorrow"
    else:
        return "on " + date_obj.strftime("%A")

# --- END OF HELPERS ---
# ***********************************************************************************
# --- WEATHER ACTION ---
class ActionCheckWeather(Action):
    def name(self) -> Text:
        return "action_check_weather"
    
    def run(self, dispatcher, tracker, domain):
        # 1. Get location
        loc = next(tracker.get_latest_entity_values("GPE"), None)
        
        if not loc:
            loc = tracker.get_slot("weather_location")
            if loc and len(loc.split()) > 4:
                loc = None
        
        if loc and len(loc.split()) > 4:
            loc = None

        if not loc:
            dispatcher.utter_message(text="Please specify a location for the weather forecast.")
            return []

        # 2. Look for "time" entity from Duckling
        date_entity = next(tracker.get_latest_entity_values("time"), None)
        logger.debug(f"Weather - time entity: {date_entity}")
        
        target_date = parse_date(date_entity) if date_entity else datetime.datetime.now()
        friendly_day = format_friendly_date(target_date)
        target_weekday = target_date.strftime("%A").lower()

        # 3. Call API
        try:
            resp = requests.post(WEATHER_API_URL, data={"place": loc})
            data = resp.json()
            
            if not data or "forecast" not in data:
                dispatcher.utter_message(text=f"I couldn't find weather data for {loc}.")
                return [SlotSet("weather_location", None), SlotSet("location", None)]

            found = next((d for d in data["forecast"] if d['day'].lower() == target_weekday), None)
            
            if found:
                desc = found.get('weather')
                temp = found.get('temperature', {}).get('max')
                dispatcher.utter_message(text=f"The weather in {loc} {friendly_day} will be {desc} with {temp} degrees.")
            else:
                dispatcher.utter_message(text=f"No forecast available for {friendly_day} in {loc}.")
                
        except Exception as e:
            logger.error(f"Weather API Error: {e}")
            dispatcher.utter_message(text="Weather service is currently unreachable.")

        return [SlotSet("weather_location", loc), SlotSet("location", None)]

# --- CALENDAR: CREATE ---
class ActionCreateAppointment(Action):
    def name(self) -> Text:
        return "action_create_appointment"

    def run(self, dispatcher, tracker, domain):
        title = tracker.get_slot("event_title")
        date_text = tracker.get_slot("target_date")
        loc = tracker.get_slot("location")

        if loc:
            loc_l = loc.lower()
            if ("appointment" in loc_l) or ("titled" in loc_l) or len(loc.split()) > 6:
                loc = None
        
        logger.debug(f"ActionCreate - title: '{title}', date: '{date_text}', loc: '{loc}'")

        if not title or title.strip() == "" or title == ".":
            dispatcher.utter_message(text="I couldn't understand the title. Please try again :)")
            return [
                SlotSet("event_title", None),
                SlotSet("target_date", None),
                SlotSet("location", None),
                SlotSet("weather_location", None),
                SlotSet("requested_slot", None)
            ]
        
        if not loc or loc.lower() in ["skip", "no","none", "no location"]:
            loc = "No Location"
        
        recovered = None
        if (not date_text) or (not looks_like_date(date_text)):
            for ev in reversed(tracker.events[-20:]):
                if ev.get("event") != "user":
                    continue
                ents = ev.get("parse_data", {}).get("entities", [])
                for ent in ents:
                    if ent.get("entity") == "time":
                        candidate = ent.get("value")
                        if looks_like_date(candidate):
                            recovered = candidate
                            break
                if recovered:
                    break

        if recovered:
            date_text = recovered
            logger.debug(f"Recovered target_date from entity: {date_text}")

        # Check if date_text is valid before parsing
        if not date_text or not looks_like_date(date_text):
            dispatcher.utter_message(text="I didn't get the date. Please try again.")
            return [
                SlotSet("target_date", None),
                SlotSet("requested_slot", "target_date")
            ]
        
        dt = parse_date(date_text)

        if not dt:
            logger.error(f"Failed to parse date '{date_text}'")
            dispatcher.utter_message(text="I couldn't understand that date. Please try again.")
            return [
                SlotSet("target_date", None),
                SlotSet("requested_slot", "target_date")
            ]
        

        start_str = dt.strftime("%Y-%m-%dT%H:%M")
        end_str = (dt + datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")

        payload = {
            "title": title,
            "description": "Voice Assistant",
            "start_time": start_str,
            "end_time": end_str,
            "location": loc
        }

        try:
            resp = requests.post(
                CALENDAR_API_URL,
                params={"calenderid": CALENDAR_ID},
                json=payload
            )
            
            logger.info(f"API Response: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                
                if "error" in data:
                    logger.error(f"API Error: {data['error']}")
                    dispatcher.utter_message(text="Failed to save appointment.")
                    return []
                
                new_id = data.get("id")
                
                if loc == "No Location":
                    try:
                        dt_formatted = dt.strftime("%B %d")  # "January 12"
                        dispatcher.utter_message(text=f"Scheduled '{title}' for {dt_formatted}.")
                    except:
                        dispatcher.utter_message(text=f"Scheduled '{title}' for {date_text}.")
                else:
                    try:
                        dt_formatted = dt.strftime("%B %d")  # "January 12"
                        dispatcher.utter_message(text=f"Scheduled '{title}' for {dt_formatted} at {loc}.")
                    except:
                        dispatcher.utter_message(text=f"Scheduled '{title}' for {date_text} at {loc}.")
                
                # To clear all form slots to prevent carryover
                return [
                    SlotSet("last_interaction_id", str(new_id)),
                    SlotSet("last_created_appointment", f"{title} on {date_text}"),
                    SlotSet("event_title", None),
                    SlotSet("target_date", None),
                    SlotSet("location", None),
                    SlotSet("weather_location", None),
                    SlotSet("requested_slot", None)
                ]
            else:
                dispatcher.utter_message(text="Failed to save the appointment. API error.")
                return []
                
        except Exception as e:
            logger.error(f"Calendar API Error: {e}")
            dispatcher.utter_message(text="I've failed to save the appointment. I am sorry.")
            return []
        
# --- CALENDAR: LIST ---
class ActionListAppointments(Action):
    def name(self) -> Text:
        return "action_list_appointments"
    
    def run(self, dispatcher, tracker, domain):
        try:
            resp = requests.get(CALENDAR_API_URL, params={'calenderid': CALENDAR_ID})
            logger.info(f"List API Response: {resp.status_code}")
            
            if resp.status_code != 200:
                dispatcher.utter_message(text="Could not access calendar.")
                return []
                
            data = resp.json()
            
            if isinstance(data, dict) and "error" in data:
                logger.error(f"API Error: {data['error']}")
                dispatcher.utter_message(text="Could not access calendar.")
                return []
            
            items = data if isinstance(data, list) else []

            if not items or len(items) == 0:
                dispatcher.utter_message(text="You have no appointments scheduled.")
                return []

            now = datetime.datetime.now().replace(tzinfo=None)
            future = []
            
            for i in items:
                try:
                    item_dt = dateparser.parse(i['start_time']).replace(tzinfo=None)
                    if item_dt > now:
                        future.append(i)
                except Exception as e:
                    logger.debug(f"Date parsing error for item: {e}")
                    continue
                
            if not future:
                dispatcher.utter_message(text="You have no upcoming appointments.")
                return []
            
            # Sort by date
            future.sort(key=lambda x: dateparser.parse(x['start_time']))
            next_app = future[0]
            
            t = next_app['title']
            dt = dateparser.parse(next_app['start_time'])
            l = next_app.get('location', 'No Location')
            
            # Use proper datetime arithmetic
            today = datetime.datetime.now().date()
            apt_date = dt.date()
            
            if apt_date == today:
                date_str = "today"
            elif apt_date == (today + datetime.timedelta(days=1)):
                date_str = "tomorrow"
            else:
                date_str = "on " + dt.strftime("%A, %B %d")
            
            now_dt = datetime.datetime.now()
            if abs((dt.time().hour * 60 + dt.time().minute) - (now_dt.time().hour * 60 + now_dt.time().minute)) <= 2:
                dt = dt.replace(hour=0, minute=0)

            if dt.hour == 0 and dt.minute == 0:
                # No specific time was set
                if l and l != "No Location":
                    dispatcher.utter_message(text=f"Next appointment: '{t}' {date_str} at {l}.")
                else:
                    dispatcher.utter_message(text=f"Next appointment: '{t}' {date_str}.")
            else:
                # Specific time was set
                time_str = dt.strftime("%I:%M %p").lstrip('0')
                if l and l != "No Location":
                    dispatcher.utter_message(text=f"Next appointment: '{t}' {date_str} at {time_str} in {l}.")
                else:
                    dispatcher.utter_message(text=f"Next appointment: '{t}' {date_str} at {time_str}.")
                    
        except Exception as e:
            logger.error(f"List Appointments Error: {e}")
            dispatcher.utter_message(text="Could not access calendar. I am sorry.")
            
        return []


# --- CALENDAR: DELETE PREVIOUS ---
class ActionDeletePrevious(Action):
    def name(self) -> Text:
        return "action_delete_previous_appointment"

    def run(self, dispatcher, tracker, domain):
        last_id = tracker.get_slot("last_interaction_id")
        if not last_id:
            dispatcher.utter_message(text="I don't know which one was created last.")
            return []

        title = "the last appointment"
        date_str = ""

        # fetch BEFORE delete
        try:
            get_resp = requests.get(
                CALENDAR_API_URL,
                params={"id": last_id, "calenderid": CALENDAR_ID},
                timeout=10,
            )
            if get_resp.status_code == 200:
                appt = get_resp.json()
                title = appt.get("title", title)
                if isinstance(appt, dict) and "start_time" in appt:
                    dt = dateparser.parse(appt["start_time"])
                    date_str = dt.strftime("%B %d")
        except Exception:
            pass

        try:
            del_resp = requests.delete(
                CALENDAR_API_URL,
                params={"id": last_id, "calenderid": CALENDAR_ID},
                timeout=10,
            )
            if del_resp.status_code == 200:
                if date_str:
                    dispatcher.utter_message(text=f"Deleted: '{title}' on {date_str}.")
                else:
                    dispatcher.utter_message(text=f"Deleted: '{title}'.")
                resp = requests.get(CALENDAR_API_URL, params={"calenderid": CALENDAR_ID}, timeout=10)
                items = resp.json() if resp.status_code == 200 else []
                if isinstance(items, list) and items:
                    items.sort(key=lambda x: int(x.get("id", 0)), reverse=True)
                    return [SlotSet("last_interaction_id", str(items[0].get("id")))]
                return [SlotSet("last_interaction_id", None)]
        except Exception:
            pass

        dispatcher.utter_message(text="Could not delete the last appointment.")
        return []
    

# --- CALENDAR: UPDATE ---
class ActionUpdateAppointment(Action):
    def name(self) -> Text:
        return "action_update_appointment"

    def run(self, dispatcher, tracker, domain):
        """Update appointment location without using forms"""

        # Step 1: Find the appointment to update
        date_entity = next(tracker.get_latest_entity_values("time"), None)

        appointment = None

        # Try to find by date first (if user said "tomorrow", "on Monday", etc.)
        if date_entity:
            try:
                parsed_dt = parse_date(date_entity)
                if parsed_dt:
                    target_date = parsed_dt.date()
                    resp = requests.get(
                        CALENDAR_API_URL,
                        params={"calenderid": CALENDAR_ID},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        items = resp.json()
                        if isinstance(items, list):
                            for item in items:
                                try:
                                    item_date = dateparser.parse(item["start_time"]).date()
                                    if item_date == target_date:
                                        appointment = item
                                        break
                                except Exception:
                                    continue
            except Exception as e:
                logger.error(f"Date search failed: {e}")

        # Fallback to last_interaction_id
        if not appointment:
            last_id = tracker.get_slot("last_interaction_id")
            if last_id:
                try:
                    resp = requests.get(
                        CALENDAR_API_URL,
                        params={"id": last_id, "calenderid": CALENDAR_ID},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict) and "error" not in data:
                            appointment = data
                except Exception as e:
                    logger.error(f"ID lookup failed: {e}")

        if not appointment:
            dispatcher.utter_message(text="I couldn't find that appointment.")
            return [SlotSet("location", None)]

        # Step 2: Get the new location (guard against polluted slot)
        new_loc = tracker.get_slot("location")
        last_msg = (tracker.latest_message.get("text") or "").strip().lower()

        # If your domain mapping filled location from full user text, ignore it
        if new_loc and new_loc.strip().lower() == last_msg:
            new_loc = None

        # Simple fallback: take latest GPE entity (works when user says "to Berlin")
        if not new_loc:
            new_loc = next(tracker.get_latest_entity_values("location"), None)

        if not new_loc:
            new_loc = next(tracker.get_latest_entity_values("GPE"), None)
            
        # Last fallback: scan a few recent user messages for a GPE entity
        if not new_loc:
            for event in reversed(tracker.events[-10:]):
                if event.get("event") == "user":
                    entities = event.get("parse_data", {}).get("entities", [])
                    for entity in entities:
                        if entity.get("entity") in ["location", "GPE"]:
                            new_loc = entity.get("value")
                            break
                    if new_loc:
                        break

        if not new_loc:
            dispatcher.utter_message(text="Where would you like to change it to?")
            return [FollowupAction("action_listen")]

        # Step 3: Update the appointment (your API: delete old + create new)
        try:
            old_loc = appointment.get("location", "No Location")
            old_id = appointment.get("id")
            title = appointment.get("title", "appointment")

            del_resp = requests.delete(
                CALENDAR_API_URL,
                params={"id": old_id, "calenderid": CALENDAR_ID},
                timeout=10,
            )
            if del_resp.status_code != 200:
                dispatcher.utter_message(text="Failed to update appointment.")
                return [SlotSet("location", None)]

            new_appointment = {
                "title": appointment.get("title", title),
                "description": appointment.get("description", "Voice Assistant"),
                "start_time": appointment.get("start_time"),
                "end_time": appointment.get("end_time"),
                "location": str(new_loc).strip(),
            }

            create_resp = requests.post(
                CALENDAR_API_URL,
                params={"calenderid": CALENDAR_ID},
                json=new_appointment,
                timeout=10,
            )

            if create_resp.status_code == 200:
                data = create_resp.json()
                new_id = data.get("id")
                dispatcher.utter_message(text=f"Changed '{title}' location from {old_loc} to {new_loc}.")
                return [
                    SlotSet("last_interaction_id", str(new_id)),
                    SlotSet("location", None),
                ]

            dispatcher.utter_message(text="Failed to update appointment.")
            return [SlotSet("location", None)]

        except requests.exceptions.Timeout:
            dispatcher.utter_message(text="Calendar service timeout. Please try again.")
            return [SlotSet("location", None)]
        except Exception as e:
            logger.error(f"Update Error: {e}")
            dispatcher.utter_message(text="Could not update appointment.")
            return [SlotSet("location", None)]

    

# --- CALENDAR: DELETE SPECIFIC ---
class ActionDeleteSpecific(Action):
    def name(self) -> Text:
        return "action_delete_appointment"
    
    def run(self, dispatcher, tracker, domain):
        title = next(tracker.get_latest_entity_values("event_title"), None)
        date_text = next(tracker.get_latest_entity_values("time"), None)
        
        try:
            resp = requests.get(CALENDAR_API_URL, params={'calenderid': CALENDAR_ID})
            items = resp.json()
            
            if isinstance(items, dict) and "error" in items:
                dispatcher.utter_message(text="Could not access calendar.")
                return []
            
            if not items:
                dispatcher.utter_message(text="No appointments to delete.")
                return []
            
            to_delete = None
            
            if title:
                to_delete = next((i for i in items if title.lower() in i['title'].lower()), None)
            elif date_text:
                target_date = parse_date(date_text).date()
                for i in items:
                    item_date = dateparser.parse(i['start_time']).date()
                    if item_date == target_date:
                        to_delete = i
                        break
            
            if to_delete:
                requests.delete(
                    CALENDAR_API_URL,
                    params={"id": to_delete['id'], "calenderid": CALENDAR_ID}
                )
                dispatcher.utter_message(text=f"Deleted appointment: {to_delete['title']}")
            else:
                dispatcher.utter_message(text="Could not find that appointment.")
                
        except Exception as e:
            logger.error(f"Delete Specific Error: {e}")
            dispatcher.utter_message(text="Could not delete appointment.")
        
        return []

# --- FORM VALIDATION ---
class ValidateCalendarForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_calendar_form"
    
    def get_valid_historical_value(self, tracker, slot_name):
        """
        Robust Time Travel: Finds the value of a slot from BEFORE the current user turn.
        It scans backwards. It only considers a slot value 'valid' if it was set
        PRIOR to the most recent 'user' event.
        """
        seen_latest_user_message = False
        
        for event in reversed(tracker.events):
            event_type = event.get("event")
            
            # We must cross the boundary of the current user input to find 'history'
            if event_type == "user":
                seen_latest_user_message = True
                continue
            
            # If we haven't seen the user message yet, we are still in the "dirty" current turn.
            if not seen_latest_user_message:
                continue

            # Now we are in safe history. If we find the slot, return it.
            if event_type == "slot" and event.get("name") == slot_name:
                return event.get("value")
        
        return None
    
    def validate_event_title(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        """ Minimalistic Validation for title """
        requested_slot = tracker.get_slot("requested_slot")

        # --- 1. STRICT GUARD: If not asking for title, protect the old one ---
        if requested_slot and requested_slot != "event_title":
            
            # A. Check if the "new" title looks like a Date. If so, it's definitely wrong.
            if looks_like_date(slot_value):
                 logger.debug(f"Rejected '{slot_value}' as title because it looks like a date.")
                 original_title = self.get_valid_historical_value(tracker, "event_title")
                 return {"event_title": original_title} if original_title else {"event_title": None}

            # B. Standard Revert Logic
            original_title = self.get_valid_historical_value(tracker, "event_title")
            
            if original_title:
                logger.debug(f"Reverting Title from '{slot_value}' back to stable '{original_title}'")
                return {"event_title": original_title}
            else:
                return {"event_title": None}
        
        if not slot_value:
            logger.debug("Title is empty/None")
            return {"event_title": None}
        
        txt = str(slot_value).strip()
        logger.debug(f"Validating title: '{txt}'")
        
        clean_regex = r"^(?:add|create|schedule|book|new|set up|make|put)\s+(?:an|a|the)?\s*(?:appointment|meeting|event|session|call|date)?\s*(?:titled|called|named|with|for)?\s+"
        
        # Perform the replacement (case insensitive)
        new_txt = re.sub(clean_regex, "", txt, flags=re.IGNORECASE).strip()
        
        if new_txt != txt:
            logger.debug(f"Cleaned title from '{txt}' to '{new_txt}'")
            txt = new_txt
        suffix_regex = r"(.*)\s+(?:for|on|at|in)\s+(.+)$"
        match = re.match(suffix_regex, txt, re.IGNORECASE)
        
        if match:
            potential_title = match.group(1).strip()
            potential_date_str = match.group(2).strip()
            
            # Check if the part after 'for/on/at' is actually a date
            if looks_like_date(potential_date_str):
                logger.debug(f"Stripped date suffix '{potential_date_str}' from title. Keeping '{potential_title}'")
                txt = potential_title

        # --- 3. FORCE LOCATION ASK ---
        logger.debug("Force-resetting Location to None to ensure User is asked.")
        
        # Check basic validity
        if not txt or txt in [".", ",", "!", "?"]:
            dispatcher.utter_message(text="I didn't catch that. What is the appointment name?")
            return {"event_title": None}
        
        if txt.lower() in ["skip", "skipudio", 'escapeio', 'no', 'none']:
            dispatcher.utter_message(text="I didn't understand. What is the appointment name again?")
            return {"event_title": None}
        
        # If the cleaned title looks like a date, reject it even if requested
        if looks_like_date(txt):
             dispatcher.utter_message(text="That sounds like a date, not a title. What is the name of the appointment?")
             return {"event_title": None}
        
        logger.debug(f"Title accepted: '{txt}'")
        return {'event_title': txt, 'location': None}
    
    def validate_target_date(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        """Validate date - reject if it's in the past"""

        requested_slot = tracker.get_slot("requested_slot")

        if requested_slot and requested_slot != "target_date":
            # Revert logic for Date
            if requested_slot and requested_slot != "target_date":
                original_date = self.get_valid_historical_value(tracker, "target_date")
                if original_date:
                    logger.debug(f"Reverting Date from '{slot_value}' back to stable '{original_date}'")
                    return {"target_date": original_date}
                else:
                    return {"target_date": None}
            
        if not slot_value:
            logger.debug("Validate_target_date: Date is empty/None")
            return {"target_date": None}
        
        txt = str(slot_value).strip()
        logger.debug(f"Validating date: '{txt}'")

        if not looks_like_date(txt):
            dispatcher.utter_message(
                 text="I was asking for the date (e.g., 'tomorrow', 'Friday', '14th of February'). When should I schedule it?"
        )
            return {"target_date": None}

        parsed = parse_date(txt)

        if not parsed or not valid_day_in_month(txt, parsed):
            dispatcher.utter_message(text="That date doesn't seem valid. Please give a real date (e.g., '14th of February').")
            return {"target_date": None}
      
        # check if the date is in the pass
        now = datetime.datetime.now()
        if parsed < now:
            logger.debug(f"Data in the past: {parsed}")
            if parsed.date() == now.date():
                dispatcher.utter_message(text="That time has already passed. Can't set an appointment.")
            else:
                dispatcher.utter_message(text="That date is in the past. When should i schedule it?")
            return {"target_date":None}
        
        logger.debug(f"Data accepted: '{txt}' -> {parsed}")
        return {"target_date": txt}
    
    
    def validate_location(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        """ Minimal validation for location"""
        if tracker.get_slot("requested_slot") != "location":
            return {}
        
        if not slot_value:
            logger.debug("validate_location: Location is empty/None")
            return {"location": None}
        
        txt = str(slot_value).strip().lower()
        bad_location_markers = ["appointment", "titled", "schedule", "add an", "add", "for the", "for tomorrow"]
        if any(m in txt for m in bad_location_markers) and len(txt.split()) > 4:
                dispatcher.utter_message(text="I was asking for the location (e.g., 'University Library', 'Room 242'). Where is it?")
                return {"location": None}

        if looks_like_date(txt):
                dispatcher.utter_message(text="I was asking for the location (e.g., 'Room 245', 'Office', 'Frankfurt'). Where is it?")
                return {"location": None}
        
        skip_phrases = ["skip", "no", "nope", "none", "no location", "nowhere", "nothing", "no place"]
        if any(phrase in txt.lower() for phrase in skip_phrases):
            logger.debug("Detected skip phrase, setting to 'No Location'")
            return {"location": "No Location"}
        
        logger.debug(f"Location accepted: '{txt}'")
        return {"location": slot_value.strip()}
    
# --- DEACTIVATE LOOP ---
class ActionDeactivateLoop(Action):
    def name(self) -> Text:
        return "action_deactivate_loop"
    
    def run(self, dispatcher, tracker, domain):
        return [
            ActiveLoop(None),
            SlotSet("requested_slot", None),
            SlotSet("event_title", None),
            SlotSet("target_date", None),
            SlotSet("location", None)
        ]
    
class ActionResetFormSlots(Action):
    def name(self) -> Text:
        return "action_reset_form_slots" 

    def run(self, dispatcher, tracker, domain):
        return [
            SlotSet("event_title", None),
            SlotSet("target_date", None),
            SlotSet("location", None),
            SlotSet("requested_slot", None)
        ]
    
# For debugging slots
class ActionLogSlots(Action):
    def name(self) -> Text:
        return "action_log_slots"
    
    def run(self, dispatcher, tracker, domain):
        slots = tracker.current_slot_values()
        logger.debug("=" * 50)
        logger.debug("CURRENT SLOT VALUES:")
        for key, value in slots.items():
            if value is not None:
                logger.debug(f"  {key}: {value}")
        logger.debug("=" * 50)
        return []