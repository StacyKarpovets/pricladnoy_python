import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import io
import random

import requests
import matplotlib.pyplot as plt
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

USDA_API_KEY = os.environ.get('USDA_API_KEY')
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

if not all([USDA_API_KEY, OPENWEATHER_API_KEY, TELEGRAM_BOT_TOKEN]):
    logger.error("Не установлены обязательные переменные окружения!")
    exit(1)

def log_command(user_id: int, username: str, command: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - User {user_id} ({username}) sent command: {command}"

    with open('bot_commands.log', 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

    logger.info(log_message)

class WeatherAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        self.cache = {}

    async def get_temperature(self, city: str) -> tuple:
        """Получить температуру для города (градусы Цельсия, успех)"""
        city_lower = city.lower().strip()

        logger.info(f"🌤️ Запрос погоды для города: '{city}'")

        if city_lower in self.cache:
            cached_temp, cached_time = self.cache[city_lower]
            if datetime.now() - cached_time < timedelta(minutes=30):
                logger.info(f"Используем кэш для города: {city} = {cached_temp}°C")
                return cached_temp, True

        temperature = await self._try_api_requests(city)
        if temperature is not None:
            self.cache[city_lower] = (temperature, datetime.now())
            return temperature, True

        temperature = await self._try_geocoding(city)
        if temperature is not None:
            self.cache[city_lower] = (temperature, datetime.now())
            return temperature, True

        logger.warning(f"Не удалось получить погоду для города: {city}")
        return 20.0, False

    async def _try_api_requests(self, city: str) -> Optional[float]:
        query_variants = [
            f"{city},RU",
            f"{city},Russia",
            city,
            f"{city.title()},RU",
            f"{city.lower()},ru"
        ]

        for query in query_variants:
            try:
                params = {
                    'q': query,
                    'appid': self.api_key,
                    'units': 'metric',
                    'lang': 'ru'
                }

                logger.info(f"Пробуем API запрос: q={query}")
                response = requests.get(self.base_url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    temperature = data['main']['temp']
                    city_name = data.get('name', city)
                    logger.info(f"API успешно: {city_name} = {temperature}°C")
                    return temperature
                elif response.status_code == 404:
                    logger.warning(f"Город не найден в API: {query}")
                elif response.status_code == 401:
                    logger.error("Ошибка 401: Неверный API ключ OpenWeatherMap")
                    return None
                else:
                    logger.warning(f"API ошибка {response.status_code} для {query}")

            except requests.exceptions.Timeout:
                logger.warning(f"Таймаут для {query}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ошибка сети для {query}: {e}")
            except Exception as e:
                logger.error(f"Ошибка запроса для {query}: {e}")

        return None

    async def _try_geocoding(self, city: str) -> Optional[float]:
        try:
            geo_queries = [
                f"{city},RU",
                f"{city},Russia",
                city
            ]

            for query in geo_queries:
                try:
                    params = {
                        'q': query,
                        'limit': 5,
                        'appid': self.api_key
                    }

                    logger.info(f"Пробуем геокодинг: q={query}")
                    response = requests.get(self.geo_url, params=params, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            location = data[0]
                            lat = location['lat']
                            lon = location['lon']
                            city_name = location.get('name', city)

                            logger.info(f"Координаты найдены: {city_name} ({lat}, {lon})")

                            weather_params = {
                                'lat': lat,
                                'lon': lon,
                                'appid': self.api_key,
                                'units': 'metric',
                                'lang': 'ru'
                            }

                            weather_response = requests.get(
                                self.base_url,
                                params=weather_params,
                                timeout=10
                            )

                            if weather_response.status_code == 200:
                                weather_data = weather_response.json()
                                temperature = weather_data['main']['temp']
                                weather_city = weather_data.get('name', city_name)
                                logger.info(f"Погода по координатам: {weather_city} = {temperature}°C")
                                return temperature

                except requests.exceptions.Timeout:
                    logger.warning(f"⏱Таймаут геокодинга для {query}")
                except Exception as e:
                    logger.warning(f"Ошибка геокодинга для {query}: {e}")

        except Exception as e:
            logger.error(f"Критическая ошибка геокодинга: {e}")

        return None

weather_api = WeatherAPI(OPENWEATHER_API_KEY)

async def get_weather(city: str) -> tuple:
    return await weather_api.get_temperature(city)

profile_temp_data = {}

@dataclass
class UserProfile:
    user_id: int
    weight: float
    height: float
    age: int
    gender: str  # 'male' or 'female'
    activity_level: str  # 'sedentary', 'light', 'moderate', 'active', 'very_active'
    city: str
    goal: str  # 'lose', 'maintain', 'gain'
    water_goal: int = 0
    calorie_goal: int = 0
    protein_goal: int = 0
    fat_goal: int = 0
    carbs_goal: int = 0

@dataclass
class DailyLog:
    date: str
    water_consumed: float = 0
    calories_consumed: float = 0
    calories_burned: float = 0
    foods: List[Dict] = None
    workouts: List[Dict] = None

    def __post_init__(self):
        if self.foods is None:
            self.foods = []
        if self.workouts is None:
            self.workouts = []

user_profiles: Dict[int, UserProfile] = {}
user_logs: Dict[int, Dict[str, DailyLog]] = {}

WORKOUT_INTENSITY = {
    'йога': 'low',
    'пилатес': 'low',
    'растяжка': 'low',
    'ходьба': 'low',
    'прогулка': 'low',

    'силовая': 'medium',
    'тренажеры': 'medium',
    'функциональная': 'medium',
    'кардио': 'medium',
    'аэробика': 'medium',
    'танцы': 'medium',

    'бег': 'high',
    'интервальная': 'high',
    'кроссфит': 'high',
    'плавание': 'high',
    'велосипед': 'high',
    'скакалка': 'high',
    'бокс': 'high'
}

COMMON_FOODS_DB = {
    'банан': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3},
    'яблоко': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2},
    'апельсин': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1},
    'яблоки': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2},

    'картофель': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1},
    'морковь': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2},
    'помидор': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2},
    'помидоры': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2},
    'огурец': {'calories': 15, 'protein': 0.7, 'carbs': 3.6, 'fat': 0.1},
    'огурцы': {'calories': 15, 'protein': 0.7, 'carbs': 3.6, 'fat': 0.1},

    'курица': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
    'куриный': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
    'грудка': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
    'яйцо': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11},
    'яйца': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11},
    'творог': {'calories': 159, 'protein': 18, 'carbs': 3.4, 'fat': 9},

    'молоко': {'calories': 42, 'protein': 3.4, 'carbs': 4.8, 'fat': 1},
    'сыр': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33},
    'йогурт': {'calories': 59, 'protein': 3.5, 'carbs': 4.7, 'fat': 3.3},

    'рис': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
    'гречка': {'calories': 343, 'protein': 13, 'carbs': 72, 'fat': 3.4},
    'овсянка': {'calories': 389, 'protein': 17, 'carbs': 66, 'fat': 7},
    'хлеб': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2},

    'вода': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'кофе': {'calories': 3.5, 'protein': 0, 'carbs': 0.2, 'fat': 0},
    'чай': {'calories': 5, 'protein': 0, 'carbs': 0.3, 'fat': 0},
}

class USDAFoodAPI:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.cache = {}

    def search_food(self, query: str) -> Optional[Dict[str, Any]]:

        cache_key = query.lower()
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=24):
                logger.info(f"Используем кэш для: {query}")
                return cached_data

        try:
            usda_result = self._search_usda_api(query)
            if usda_result and usda_result.get('calories', 0) > 0:
                usda_result['source'] = 'USDA'
                usda_result['confidence'] = 'high'
                self.cache[cache_key] = (usda_result, datetime.now())
                return usda_result

            local_result = self._search_local_db(query)
            local_result['source'] = 'Local DB'
            local_result['confidence'] = 'medium' if local_result.get('is_common', False) else 'low'
            self.cache[cache_key] = (local_result, datetime.now())
            return local_result

        except Exception as e:
            logger.error(f"Ошибка при поиске продукта '{query}': {e}")
            return self._search_local_db(query)

    def _search_usda_api(self, query: str) -> Optional[Dict[str, Any]]:
        """Поиск через USDA FoodData Central API"""
        try:
            url = f"{self.base_url}/foods/search"
            params = {
                'api_key': self.api_key,
                'query': query,
                'pageSize': 5,
                'dataType': ["Foundation", "SR Legacy", "Survey (FNDDS)"],
                'sortBy': 'dataType.keyword'
            }

            logger.info(f"Запрос к USDA API: {query}")
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                total_hits = data.get('totalHits', 0)
                logger.info(f"USDA нашёл {total_hits} результатов для '{query}'")

                if total_hits > 0:
                    foods = data.get('foods', [])

                    best_food = None
                    best_score = 0

                    for food in foods:
                        score = self._calculate_relevance_score(food, query)
                        if score > best_score:
                            best_score = score
                            best_food = food

                    if best_food:
                        return self._extract_nutrients(best_food)

            elif response.status_code == 403:
                logger.error("Доступ к USDA API запрещен. Проверьте API ключ.")
            elif response.status_code == 429:
                logger.warning("Превышен лимит запросов USDA API")
            else:
                logger.error(f"USDA API ошибка {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            logger.error("Таймаут при запросе к USDA API")
        except Exception as e:
            logger.error(f"Ошибка USDA API: {e}")

        return None

    def _calculate_relevance_score(self, food: Dict, query: str) -> int:
        """Оценка релевантности продукта"""
        score = 0
        description = food.get('description', '').lower()
        query_lower = query.lower()

        if query_lower in description:
            score += 100

        if any(word in description for word in query_lower.split()):
            score += 50

        if food.get('dataType') == 'Foundation':
            score += 30

        if food.get('foodNutrients'):
            nutrient_count = len(food['foodNutrients'])
            if nutrient_count > 10:
                score += 20

        return score

    def _extract_nutrients(self, food: Dict) -> Dict[str, Any]:
        nutrients = {}

        for nutrient in food.get('foodNutrients', []):
            nutrient_name = nutrient.get('nutrientName')
            if nutrient_name:
                nutrients[nutrient_name] = nutrient.get('value', 0)

        result = {
            'name': food.get('description', 'Unknown'),
            'calories': nutrients.get('Energy', 0),
            'protein': nutrients.get('Protein', 0),
            'carbs': nutrients.get('Carbohydrate, by difference', 0),
            'fat': nutrients.get('Total lipid (fat)', 0),
            'serving_size': 100,
            'serving_unit': 'g',
            'data_type': food.get('dataType', 'Unknown')
        }

        if food.get('brandOwner'):
            result['brand'] = food['brandOwner']

        return result

    def _search_local_db(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        if query_lower in COMMON_FOODS_DB:
            return {
                'name': query,
                **COMMON_FOODS_DB[query_lower],
                'is_common': True
            }

        for key, value in COMMON_FOODS_DB.items():
            if key in query_lower:
                return {
                    'name': query,
                    **value,
                    'is_common': True
                }

        category_mappings = {
            'суп': {'calories': 50, 'protein': 3, 'carbs': 7, 'fat': 1.5},
            'салат': {'calories': 35, 'protein': 2, 'carbs': 5, 'fat': 1},
            'пицц': {'calories': 285, 'protein': 12, 'carbs': 33, 'fat': 11},
            'бургер': {'calories': 295, 'protein': 17, 'carbs': 24, 'fat': 14},
        }

        for category, values in category_mappings.items():
            if category in query_lower:
                return {
                    'name': query,
                    **values,
                    'is_common': True
                }

        return {
            'name': query,
            'calories': 200,
            'protein': 10,
            'carbs': 20,
            'fat': 8,
            'is_common': False
        }

usda_api = USDAFoodAPI(USDA_API_KEY)

class Calculator:
    """Класс для всех расчетов"""

    @staticmethod
    def calculate_tdee(profile: UserProfile) -> float:
        if profile.gender == 'male':
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
        else:
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161

        activity_factors = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }

        return bmr * activity_factors.get(profile.activity_level, 1.2)

    @staticmethod
    def calculate_goals(profile: UserProfile, weather_temp: float) -> None:

        tdee = Calculator.calculate_tdee(profile)

        if profile.goal == 'lose':
            profile.calorie_goal = int(tdee * 0.85)
        elif profile.goal == 'gain':
            profile.calorie_goal = int(tdee * 1.15)
        else:
            profile.calorie_goal = int(tdee)

        if profile.goal == 'lose':
            protein_pct, fat_pct, carbs_pct = 0.35, 0.25, 0.40
        elif profile.goal == 'gain':
            protein_pct, fat_pct, carbs_pct = 0.30, 0.25, 0.45
        else:
            protein_pct, fat_pct, carbs_pct = 0.30, 0.25, 0.45

        profile.protein_goal = int((profile.calorie_goal * protein_pct) / 4)
        profile.fat_goal = int((profile.calorie_goal * fat_pct) / 9)
        profile.carbs_goal = int((profile.calorie_goal * carbs_pct) / 4)

        base_water = profile.weight * 30

        activity_multiplier = {
            'sedentary': 1.0,
            'light': 1.2,
            'moderate': 1.4,
            'active': 1.6,
            'very_active': 1.8
        }

        water_multiplier = activity_multiplier.get(profile.activity_level, 1.0)
        profile.water_goal = int(base_water * water_multiplier)

        if weather_temp > 25:
            profile.water_goal = int(profile.water_goal * 1.2)

    @staticmethod
    def calculate_workout_calories(workout_type: str, duration: int, weight: float) -> int:
        intensity = WORKOUT_INTENSITY.get(workout_type.lower(), 'medium')

        met_values = {
            'low': 3.5,
            'medium': 6.0,
            'high': 8.5
        }

        met = met_values[intensity]
        calories = int(met * weight * (duration / 60))

        if weight > 100:
            calories = int(calories * 1.1)
        elif weight < 60:
            calories = int(calories * 0.9)

        return calories

class ChartGenerator:

    @staticmethod
    def create_progress_chart(user_id: int, days: int = 7) -> Optional[bytes]:
        from datetime import datetime, timedelta

        if user_id not in user_logs:
            return None

        dates = []
        water_data = []
        calorie_data = []
        burned_data = []

        today = datetime.now().date()
        for i in range(days):
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            dates.insert(0, date)

            if date in user_logs.get(user_id, {}):
                log = user_logs[user_id][date]
                water_data.insert(0, log.water_consumed)
                calorie_data.insert(0, log.calories_consumed)
                burned_data.insert(0, log.calories_burned)
            else:
                water_data.insert(0, 0)
                calorie_data.insert(0, 0)
                burned_data.insert(0, 0)

        if not dates:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        bars = ax1.bar(dates, water_data, color='lightblue', alpha=0.7)
        if user_id in user_profiles:
            water_goal = user_profiles[user_id].water_goal
            ax1.axhline(y=water_goal, color='red', linestyle='--', label=f'Цель: {water_goal} мл')
        ax1.set_title('Потребление воды (мл)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Мл')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        x = np.arange(len(dates))
        width = 0.35

        bars1 = ax2.bar(x - width/2, calorie_data, width, label='Потреблено', color='lightgreen', alpha=0.7)
        bars2 = ax2.bar(x + width/2, burned_data, width, label='Сожжено', color='salmon', alpha=0.7)

        if user_id in user_profiles:
            calorie_goal = user_profiles[user_id].calorie_goal
            ax2.axhline(y=calorie_goal, color='red', linestyle='--', label=f'Цель: {calorie_goal} ккал')

        ax2.set_title('Баланс калорий', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ккал')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dates, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()

    @staticmethod
    def create_macros_chart(user_id: int) -> Optional[bytes]:
        today = datetime.now().strftime('%Y-%m-%d')

        if user_id not in user_logs or today not in user_logs[user_id]:
            return None

        today_log = user_logs[user_id][today]

        total_protein = sum(food.get('protein', 0) for food in today_log.foods)
        total_carbs = sum(food.get('carbs', 0) for food in today_log.foods)
        total_fat = sum(food.get('fat', 0) for food in today_log.foods)

        if total_protein == 0 and total_carbs == 0 and total_fat == 0:
            return None

        labels = ['Белки', 'Углеводы', 'Жиры']
        sizes = [total_protein, total_carbs, total_fat]
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Распределение макронутриентов', fontsize=14, fontweight='bold')

        if user_id in user_profiles:
            profile = user_profiles[user_id]
            goals = [profile.protein_goal, profile.carbs_goal, profile.fat_goal]
            actual = sizes

            x = np.arange(len(labels))
            width = 0.35

            ax2.bar(x - width/2, goals, width, label='Цели', color='lightgray', alpha=0.7)
            ax2.bar(x + width/2, actual, width, label='Факт', color=colors, alpha=0.7)

            ax2.set_xlabel('Макронутриенты')
            ax2.set_ylabel('Граммы')
            ax2.set_title('Цели vs Факт', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()

class RecommendationEngine:

    LOW_CALORIE_FOODS = [
        {"name": "Огурец", "calories": 15, "protein": 0.7, "carbs": 3.6, "fat": 0.1},
        {"name": "Сельдерей", "calories": 14, "protein": 0.7, "carbs": 3.0, "fat": 0.2},
        {"name": "Помидор", "calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
        {"name": "Брокколи", "calories": 34, "protein": 2.8, "carbs": 7.0, "fat": 0.4},
        {"name": "Шпинат", "calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4},
        {"name": "Грейпфрут", "calories": 42, "protein": 0.8, "carbs": 10.7, "fat": 0.1},
        {"name": "Арбуз", "calories": 30, "protein": 0.6, "carbs": 7.6, "fat": 0.2},
        {"name": "Куриная грудка", "calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6},
        {"name": "Тунец", "calories": 116, "protein": 25.0, "carbs": 0.0, "fat": 0.8},
        {"name": "Яичный белок", "calories": 52, "protein": 11.0, "carbs": 0.7, "fat": 0.2},
    ]

    WORKOUT_RECOMMENDATIONS = {
        'beginner': [
            {"type": "Ходьба", "duration": 30, "calories": 150, "description": "Быстрая ходьба в парке"},
            {"type": "Йога", "duration": 30, "calories": 120, "description": "Утренняя йога для бодрости"},
            {"type": "Плавание", "duration": 30, "calories": 250, "description": "Легкое плавание в бассейне"},
        ],
        'intermediate': [
            {"type": "Бег", "duration": 30, "calories": 300, "description": "Бег трусцой в среднем темпе"},
            {"type": "Велосипед", "duration": 45, "calories": 350, "description": "Езда на велосипеде по городу"},
            {"type": "Кардио", "duration": 40, "calories": 280, "description": "Кардио-тренировка с упражнениями"},
        ],
        'advanced': [
            {"type": "Интервальная", "duration": 30, "calories": 400, "description": "HIIT тренировка высокой интенсивности"},
            {"type": "Кроссфит", "duration": 45, "calories": 500, "description": "Функциональная тренировка на все группы мышц"},
            {"type": "Бег", "duration": 60, "calories": 600, "description": "Интервальный бег с ускорениями"},
        ]
    }

    @staticmethod
    def get_food_recommendations(user_id: int, calorie_budget: float) -> List[Dict]:
        if user_id not in user_profiles:
            return []

        profile = user_profiles[user_id]

        suitable_foods = []
        for food in RecommendationEngine.LOW_CALORIE_FOODS:
            food_calories = food['calories']

            if profile.goal == 'lose':
                if food_calories <= 150:
                    suitable_foods.append(food)
            elif profile.goal == 'gain':
                if food_calories >= 200:
                    suitable_foods.append(food)
            else:
                if 100 <= food_calories <= 300:
                    suitable_foods.append(food)

        if len(suitable_foods) > 3:
            recommendations = random.sample(suitable_foods, 3)
        else:
            recommendations = suitable_foods

        return recommendations

    @staticmethod
    def get_workout_recommendations(user_id: int) -> List[Dict]:
        if user_id not in user_profiles:
            return RecommendationEngine.WORKOUT_RECOMMENDATIONS['beginner']

        profile = user_profiles[user_id]

        activity_levels = {
            'sedentary': 'beginner',
            'light': 'beginner',
            'moderate': 'intermediate',
            'active': 'intermediate',
            'very_active': 'advanced'
        }

        level = activity_levels.get(profile.activity_level, 'beginner')
        return RecommendationEngine.WORKOUT_RECOMMENDATIONS[level]

    @staticmethod
    def get_daily_tips(user_id: int) -> str:
        if user_id not in user_profiles:
            return "Совет: Начните с настройки профиля для персонализированных рекомендаций!"

        profile = user_profiles[user_id]
        today = datetime.now().strftime('%Y-%m-%d')

        tips = []

        if user_id in user_logs and today in user_logs[user_id]:
            water_consumed = user_logs[user_id][today].water_consumed
            water_percent = (water_consumed / profile.water_goal) * 100

            if water_percent < 50:
                tips.append("💧 Вы еще не достигли половины нормы воды. Не забывайте пить регулярно!")
            elif water_percent < 80:
                tips.append("💧 Отличный прогресс по воде! Еще немного до цели.")

        if profile.goal == 'lose':
            tips.append("🎯 Для похудения: старайтесь ужинать за 3-4 часа до сна.")
            tips.append("🍎 Замените перекусы на фрукты или овощи.")
        elif profile.goal == 'gain':
            tips.append("🎯 Для набора массы: добавьте белковые продукты в каждый прием пищи, ешьте больше углеводов.")

        general_tips = [
            "⏰ Старайтесь есть в одно и то же время каждый день.",
            "🧘‍♀️ Не забывайте про растяжку после тренировок.",
            "💤 Здоровый сон (7-9 часов) важен для восстановления.",
            "📱 Отмечайте в боте каждый прием пищи.",
            "🥗 Добавьте больше овощей в ваш рацион.",
            "🚶‍♂️ Делайте короткие перерывы на прогулку каждый час."
        ]

        if len(tips) < 3:
            tips.extend(random.sample(general_tips, 3 - len(tips)))

        return "\n".join(tips[:3])


async def handle_profile_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = update.message.text.strip()

    logger.info(f"Получено сообщение от пользователя {user_id}: {text}")

    parts = text.split()
    if len(parts) != 4:
        logger.info(f"Сообщение не является данными профиля (не 4 части): {text}")
        return

    try:
        weight = float(parts[0])
        height = float(parts[1])
        age = int(parts[2])
        city = parts[3]

        if not (30 <= weight <= 200):
            await update.message.reply_text("❌ Вес должен быть от 30 до 200 кг")
            return
        if not (100 <= height <= 250):
            await update.message.reply_text("❌ Рост должен быть от 100 до 250 см")
            return
        if not (10 <= age <= 120):
            await update.message.reply_text("❌ Возраст должен быть от 10 до 120 лет")
            return

    except ValueError:
        logger.info(f"Не удалось преобразовать данные профиля: {text}")
        return

    logger.info(f"Обрабатываем данные профиля для пользователя {user_id}")

    gender = None
    activity_level = None
    goal = None

    if user_id in profile_temp_data:
        user_data = profile_temp_data[user_id]
        gender = user_data.get('gender')
        activity_level = user_data.get('activity_level')
        goal = user_data.get('goal')
        logger.info(f"Данные из profile_temp_data: gender={gender}, activity={activity_level}, goal={goal}")

    if not all([gender, activity_level, goal]):
        gender = context.user_data.get('gender')
        activity_level = context.user_data.get('activity_level')
        goal = context.user_data.get('goal')
        logger.info(f"Данные из context.user_data: gender={gender}, activity={activity_level}, goal={goal}")

    if not gender:
        gender = 'male'
        logger.warning(f"Пол не найден, использую значение по умолчанию: {gender}")

    if not activity_level:
        activity_level = 'moderate'
        logger.warning(f"Уровень активности не найден, использую значение по умолчанию: {activity_level}")

    if not goal:
        goal = 'maintain'
        logger.warning(f"Цель не найдена, использую значение по умолчанию: {goal}")

    loading_msg = await update.message.reply_text(f"🌤️ Получаю погоду для {city}...")

    try:
        temp, success = await get_weather(city)

        if success:
            await loading_msg.edit_text(f"✅ Получена погода для {city}: {temp}°C")
        else:
            await loading_msg.edit_text(f"Не удалось получить погоду для {city}. Использую стандартную температуру {temp}°C.")

    except Exception as e:
        logger.error(f"Ошибка получения погоды для {city}: {e}")
        temp = 20.0
        await loading_msg.edit_text(f"Ошибка получения погоды. Использую стандартную температуру {temp}°C.")

    try:
        profile = UserProfile(
            user_id=user_id,
            weight=weight,
            height=height,
            age=age,
            gender=gender,
            activity_level=activity_level,
            city=city,
            goal=goal
        )

        Calculator.calculate_goals(profile, temp)

        user_profiles[user_id] = profile

        today = datetime.now().strftime('%Y-%m-%d')
        if user_id not in user_logs:
            user_logs[user_id] = {}
        if today not in user_logs[user_id]:
            user_logs[user_id][today] = DailyLog(date=today)

        response = f"""
        ✅ *Профиль создан успешно!*

        📊 *Ваши данные:*
        • 👤 Пол: {'Мужской ♂️' if profile.gender == 'male' else 'Женский ♀️'}
        • ⚖️ Вес: {profile.weight} кг
        • 📏 Рост: {profile.height} см
        • 🎂 Возраст: {profile.age} лет
        • 🏃 Активность: {profile.activity_level}
        • 🏙️ Город: {profile.city}
        • 🎯 Цель: {'Похудеть ⬇️' if profile.goal == 'lose' else 'Поддерживать вес ⏸️' if profile.goal == 'maintain' else 'Набрать массу ⬆️'}

        🎯 *Ваши дневные цели:*
        • 💧 Вода: {profile.water_goal} мл
        • 🔥 Калории: {profile.calorie_goal} ккал
        • 🥚 Белки: {profile.protein_goal} г
        • 🥑 Жиры: {profile.fat_goal} г
        • 🍚 Углеводы: {profile.carbs_goal} г

        🌡️ *Погода в {city}:* {temp}°C

        🚀 *Начните отслеживать:*
        • /logwater 500 - выпили воду
        • /logfood яблоко - съели еду
        • /logworkout бег 30 - завершили тренировку
        • /progress - посмотреть графики
        """

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

        if user_id in profile_temp_data:
            del profile_temp_data[user_id]

        for key in ['gender', 'activity_level', 'goal']:
            if key in context.user_data:
                del context.user_data[key]

        logger.info(f"Профиль успешно создан для пользователя {user_id}")

    except Exception as e:
        logger.error(f"Ошибка при создании профиля: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Произошла ошибка при создании профиля: {str(e)[:100]}\n"
            "Пожалуйста, попробуйте еще раз или используйте /help для справки."
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    log_command(user.id, user.username or "no username", "/start")

    user_id = update.effective_user.id

    welcome_text = """
🏋️‍♂️ *Добро пожаловать в Fitness Tracker Pro Bot\!*

✨ *Премиум\-функции:*
• 📊 Детальные графики прогресса
• 🎯 Рекомендации по питанию и тренировкам
• 🥗 Умный поиск калорийности через USDA API
• 📈 Анализ макронутриентов
• 💡 Персонализированные советы
• 🌤️ Учет погоды для нормы воды

*Основные команды:*
/profile \- Настроить/показать профиль
/editprofile \- Изменить параметры профиля
/logwater \<мл\> \- Записать воду
/logfood \<продукт\> \- Записать еду
/logworkout \<тип\> \<мин\> \- Записать тренировку
/progress \[дни\] \- Графики прогресса \(по умолчанию 7 дней\)
/macros \- Диаграмма макронутриентов
/recommend \- Рекомендации
/tips \- Советы на день
/stats \- Статистика
/reset \- Сбросить все данные
/help \- Справка

*Примеры:*
• /logwater 500
• /logfood куриная грудка
• /logworkout бег 45
• /progress 14
• /editprofile город Москва

🚀 *Для начала используйте* /profile
"""

    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN_V2)

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    log_command(user.id, user.username or "no username", "/profile")

    user_id = update.effective_user.id

    if user_id in user_profiles:
        profile = user_profiles[user_id]

        response = f"""
        📋 *Ваш профиль:*

        • 👤 Пол: {'Мужской ♂️' if profile.gender == 'male' else 'Женский ♀️'}
        • ⚖️ Вес: {profile.weight} кг
        • 📏 Рост: {profile.height} см
        • 🎂 Возраст: {profile.age} лет
        • 🏃 Активность: {profile.activity_level}
        • 🏙️ Город: {profile.city}
        • 🎯 Цель: {'Похудеть ⬇️' if profile.goal == 'lose' else 'Поддерживать вес ⏸️' if profile.goal == 'maintain' else 'Набрать массу ⬆️'}

        • 💧 Норма воды: {profile.water_goal} мл
        • 🔥 Норма калорий: {profile.calorie_goal} ккал
        • 🥚 Белки: {profile.protein_goal} г
        • 🥑 Жиры: {profile.fat_goal} г
        • 🍚 Углеводы: {profile.carbs_goal} г

        ✏️ Для изменения параметров используйте /editprofile
        """

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        return

    if user_id in profile_temp_data:
        del profile_temp_data[user_id]

    for key in ['gender', 'activity_level', 'goal']:
        if key in context.user_data:
            del context.user_data[key]

    keyboard = [
        [InlineKeyboardButton("Мужской ♂️", callback_data="gender_male")],
        [InlineKeyboardButton("Женский ♀️", callback_data="gender_female")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "👤 Выберите ваш пол:",
        reply_markup=reply_markup
    )

async def logwater_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/logwater"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    if not context.args:
        await update.message.reply_text("Использование: /logwater <количество в мл>\nПример: /logwater 500")
        return

    try:
        amount = float(context.args[0])
        if amount <= 0 or amount > 5000:
            await update.message.reply_text("Количество воды должно быть от 1 до 5000 мл")
            return

        today = datetime.now().strftime('%Y-%m-%d')
        if user_id not in user_logs:
            user_logs[user_id] = {}
        if today not in user_logs[user_id]:
            user_logs[user_id][today] = DailyLog(date=today)

        user_logs[user_id][today].water_consumed += amount

        profile = user_profiles[user_id]
        consumed = user_logs[user_id][today].water_consumed
        remaining = max(0, profile.water_goal - consumed)
        percentage = min(100, (consumed / profile.water_goal) * 100)

        progress_bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))

        response = f"""
        💧 *Вода записана!*

        • Добавлено: {amount} мл
        • Всего сегодня: {consumed:.0f} мл
        • Осталось: {remaining:.0f} мл
        • Цель: {profile.water_goal} мл

        Прогресс: {percentage:.0f}%
        {progress_bar}

        💡 Совет: Старайтесь пить воду небольшими порциями в течение дня.
        """

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число для количества воды.")

async def logfood_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/logfood"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    if not context.args:
        await update.message.reply_text(
            "Использование: /logfood <название продукта> [количество в граммах]\n"
            "Пример: /logfood банан 150"
        )
        return

    args = context.args
    if len(args) > 1 and args[-1].replace('.', '').isdigit():
        quantity = float(args[-1])
        product_name = " ".join(args[:-1])
    else:
        quantity = 100
        product_name = " ".join(args)

    loading_msg = await update.message.reply_text(
        f"🔍 *Ищу '{product_name}' в USDA базе данных...*",
        parse_mode='Markdown'
    )

    food_info = usda_api.search_food(product_name)

    if not food_info:
        await loading_msg.edit_text(
            f"❌ *Не удалось найти информацию о '{product_name}'*\n\n"
            "Попробуйте:\n"
            "• Использовать английское название (например, 'banana')\n"
            "• Указать более конкретное название",
            parse_mode='Markdown'
        )
        return

    serving_size = food_info.get('serving_size', 100)
    multiplier = quantity / serving_size

    calories = food_info['calories'] * multiplier
    protein = food_info.get('protein', 0) * multiplier
    carbs = food_info.get('carbs', 0) * multiplier
    fat = food_info.get('fat', 0) * multiplier

    today = datetime.now().strftime('%Y-%m-%d')
    if user_id not in user_logs:
        user_logs[user_id] = {}
    if today not in user_logs[user_id]:
        user_logs[user_id][today] = DailyLog(date=today)

    food_entry = {
        'name': food_info['name'],
        'quantity': quantity,
        'calories': calories,
        'protein': protein,
        'carbs': carbs,
        'fat': fat,
        'time': datetime.now().strftime('%H:%M'),
        'source': food_info.get('source', 'unknown'),
        'confidence': food_info.get('confidence', 'low')
    }

    user_logs[user_id][today].foods.append(food_entry)
    user_logs[user_id][today].calories_consumed += calories

    profile = user_profiles[user_id]
    total_calories = user_logs[user_id][today].calories_consumed
    calorie_remaining = max(0, profile.calorie_goal - total_calories)

    source_icons = {
        'USDA': 'USDA',
        'Local DB': '🏠 Локальная база',
        'unknown': '📊'
    }

    source = source_icons.get(food_info.get('source', 'unknown'), '📊')

    response = f"""
{source} *{food_info['name']}*

📏 *Количество:* {quantity} г
⚖️ *На основе:* {serving_size} г порция

📊 *Пищевая ценность:*
• 🔥 Калории: {calories:.1f} ккал
• 🥚 Белки: {protein:.1f} г
• 🍚 Углеводы: {carbs:.1f} г
• 🥑 Жиры: {fat:.1f} г

📈 *Сегодня всего:*
• Потреблено: {total_calories:.0f} / {profile.calorie_goal} ккал
• Осталось: {calorie_remaining:.0f} ккал

💡 *Совет:* Старайтесь есть больше овощей и фруктов!
    """

    await loading_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

async def logworkout_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/logworkout"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "Использование: /logworkout <тип тренировки> <длительность в минутах>\n\n"
            "Примеры типов: бег, ходьба, велосипед, плавание, йога, силовая, кардио\n"
            "Пример: /logworkout бег 45"
        )
        return

    workout_type = context.args[0]
    try:
        duration = int(context.args[1])
        if duration <= 0 or duration > 300:
            await update.message.reply_text("Длительность должна быть от 1 до 300 минут")
            return
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число для длительности тренировки.")
        return

    profile = user_profiles[user_id]
    calories_burned = Calculator.calculate_workout_calories(workout_type, duration, profile.weight)

    today = datetime.now().strftime('%Y-%m-%d')
    if user_id not in user_logs:
        user_logs[user_id] = {}
    if today not in user_logs[user_id]:
        user_logs[user_id][today] = DailyLog(date=today)

    workout_entry = {
        'type': workout_type,
        'duration': duration,
        'calories': calories_burned,
        'time': datetime.now().strftime('%H:%M'),
        'intensity': WORKOUT_INTENSITY.get(workout_type.lower(), 'medium')
    }

    user_logs[user_id][today].workouts.append(workout_entry)
    user_logs[user_id][today].calories_burned += calories_burned

    total_burned = user_logs[user_id][today].calories_burned

    extra_water = duration // 30 * 200

    response = f"""
    🏋️‍♂️ *Тренировка записана!*

    • Тип: {workout_type.title()}
    • Длительность: {duration} минут
    • Интенсивность: {WORKOUT_INTENSITY.get(workout_type.lower(), 'Средняя').title()}
    • Сожжено калорий: {calories_burned} ккал

    📊 *Сегодня всего сожжено:* {total_burned} ккал

    💧 *Рекомендация по воде:*
    Для восстановления выпейте дополнительно {extra_water} мл воды.

    🎯 *Совет:*
    Для лучших результатов чередуйте разные типы тренировок.
    """

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

async def progress_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/progress"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    days = 7
    if context.args:
        try:
            days = int(context.args[0])
            days = max(1, min(days, 30))
        except ValueError:
            pass

    await update.message.reply_text(f"📊 Генерирую графики прогресса за {days} дней...")

    chart_data = ChartGenerator.create_progress_chart(user_id, days)

    if not chart_data:
        await update.message.reply_text("Недостаточно данных для построения графика. Начните отслеживать активность!")
        return

    await update.message.reply_photo(
        photo=chart_data,
        caption=f"📈 *Прогресс за последние {days} дней*\n\n"
               f"• Синий график: Потребление воды\n"
               f"• Красная линия: Цель по воде\n"
               f"• Зеленые столбцы: Потребленные калории\n"
               f"• Красные столбцы: Сожженные калории\n"
               f"• Красная линия: Цель по калориям\n\n"
               f"Используйте /stats для детальной статистики",
        parse_mode=ParseMode.MARKDOWN
    )

async def macros_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/macros"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    await update.message.reply_text("🥗 Анализирую потребление макронутриентов...")

    chart_data = ChartGenerator.create_macros_chart(user_id)

    if not chart_data:
        await update.message.reply_text("Недостаточно данных о питании. Запишите приемы пищи с помощью /logfood")
        return

    await update.message.reply_photo(
        photo=chart_data,
        caption="📊 *Распределение макронутриентов*\n\n"
               "• Левая диаграмма: Процентное соотношение БЖУ\n"
               "• Правая диаграмма: Цели vs Факт (в граммах)\n\n"
               "💡 Для лучших результатов следите за балансом белков, жиров и углеводов!",
        parse_mode=ParseMode.MARKDOWN
    )

async def recommend_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/recommend"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    profile = user_profiles[user_id]

    today = datetime.now().strftime('%Y-%m-%d')
    consumed_calories = 0
    if user_id in user_logs and today in user_logs[user_id]:
        consumed_calories = user_logs[user_id][today].calories_consumed

    remaining_calories = max(0, profile.calorie_goal - consumed_calories)

    food_recs = RecommendationEngine.get_food_recommendations(user_id, remaining_calories)

    workout_recs = RecommendationEngine.get_workout_recommendations(user_id)

    response = f"""
    🎯 *Персонализированные рекомендации*

    *🍎 Рекомендации по питанию:*
    """

    if food_recs:
        for i, food in enumerate(food_recs[:3], 1):
            response += f"\n{i}. *{food['name']}*\n"
            response += f"   • Калории: {food['calories']} ккал/100г\n"
            response += f"   • Белки: {food.get('protein', 0)}г | Углеводы: {food.get('carbs', 0)}г | Жиры: {food.get('fat', 0)}г\n"
    else:
        response += "\nНедостаточно данных для рекомендаций по питанию\n"

    response += f"\n*🏋️‍♂️ Рекомендации по тренировкам:*\n"

    for i, workout in enumerate(workout_recs[:3], 1):
        response += f"\n{i}. *{workout['type']}*\n"
        response += f"   • Длительность: {workout['duration']} минут\n"
        response += f"   • Примерно сожжет: {workout['calories']} ккал\n"
        response += f"   • {workout['description']}\n"

    response += f"\n*📊 Ваш баланс на сегодня:*\n"
    response += f"• Осталось калорий: {remaining_calories:.0f} ккал\n"
    response += f"• Цель по белкам: {profile.protein_goal} г\n"
    response += f"• Цель по жирам: {profile.fat_goal} г\n"
    response += f"• Цель по углеводам: {profile.carbs_goal} г\n"

    response += f"\n💡 Для получения советов используйте /tips"

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

async def tips_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/tips"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    tips = RecommendationEngine.get_daily_tips(user_id)

    response = f"""
    💡 *Советы на сегодня*

    {tips}

    *Еще команды:*
    • /recommend - Персонализированные рекомендации
    • /progress - Графики прогресса
    • /macros - Анализ макронутриентов
    • /stats - Детальная статистика
    """

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/stats"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    profile = user_profiles[user_id]
    today = datetime.now().strftime('%Y-%m-%d')

    if user_id in user_logs and today in user_logs[user_id]:
        today_log = user_logs[user_id][today]

        total_protein = sum(food.get('protein', 0) for food in today_log.foods)
        total_carbs = sum(food.get('carbs', 0) for food in today_log.foods)
        total_fat = sum(food.get('fat', 0) for food in today_log.foods)

        protein_pct = (total_protein / profile.protein_goal * 100) if profile.protein_goal > 0 else 0
        carbs_pct = (total_carbs / profile.carbs_goal * 100) if profile.carbs_goal > 0 else 0
        fat_pct = (total_fat / profile.fat_goal * 100) if profile.fat_goal > 0 else 0
        water_pct = (today_log.water_consumed / profile.water_goal * 100) if profile.water_goal > 0 else 0
        calorie_pct = (today_log.calories_consumed / profile.calorie_goal * 100) if profile.calorie_goal > 0 else 0

        def create_progress_bar(percentage):
            filled = int(percentage / 10)
            return "█" * filled + "░" * (10 - filled)

        response = f"""
        📈 *Детальная статистика за сегодня*

        *💧 Вода:*
        • Выпито: {today_log.water_consumed:.0f} мл из {profile.water_goal} мл
        • Прогресс: {water_pct:.1f}%
        • {create_progress_bar(water_pct)}

        *🔥 Калории:*
        • Потреблено: {today_log.calories_consumed:.0f} ккал из {profile.calorie_goal} ккал
        • Сожжено: {today_log.calories_burned:.0f} ккал
        • Чистый баланс: {today_log.calories_consumed - today_log.calories_burned:.0f} ккал
        • Прогресс: {calorie_pct:.1f}%
        • {create_progress_bar(calorie_pct)}

        *🥗 Макронутриенты:*
        • Белки: {total_protein:.1f} г из {profile.protein_goal} г ({protein_pct:.1f}%)
        • Углеводы: {total_carbs:.1f} г из {profile.carbs_goal} г ({carbs_pct:.1f}%)
        • Жиры: {total_fat:.1f} г из {profile.fat_goal} г ({fat_pct:.1f}%)

        *🍽 Приемы пищи:* {len(today_log.foods)}
        *🏋️‍♂️ Тренировки:* {len(today_log.workouts)}

        *📊 Активность за день:*
        """

        if today_log.workouts:
            for i, workout in enumerate(today_log.workouts, 1):
                response += f"\n{i}. {workout['type'].title()} - {workout['duration']} мин ({workout['calories']} ккал)"
        else:
            response += "\nЕще не было тренировок сегодня"

        response += f"\n\n💡 Используйте /macros для визуализации данных"

    else:
        response = """
        📈 *Детальная статистика*

        Нет данных за сегодня. Начните отслеживать:

        • /logwater <мл> - запишите потребление воды
        • /logfood <продукт> - запишите прием пищи
        • /logworkout <тип> <мин> - запишите тренировку

        После накопления данных здесь появится детальная статистика!
        """

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

async def editprofile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    command_text = update.message.text if update.message else "/editprofile"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text("Сначала создайте профиль: /profile")
        return

    if not context.args:
        await update.message.reply_text(
            "Использование: /editprofile <параметр> <значение>\n\n"
            "*Доступные параметры:*\n"
            "• `город` Самара\n"
            "• `вес` 75\n"
            "• `рост` 180\n"
            "• `возраст` 25\n"
            "• `активность` moderate (sedentary, light, moderate, active, very_active)\n"
            "• `цель` lose (lose, maintain, gain)\n\n"
            "*Примеры:*\n"
            "• /editprofile город Самара\n"
            "• /editprofile вес 75\n"
            "• /editprofile активность moderate",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    param = context.args[0].lower()
    value = " ".join(context.args[1:])

    profile = user_profiles[user_id]

    try:
        old_value = getattr(profile, param, None)
        weather_temp = 20

        if param == 'город':
            profile.city = value
            temp, success = await get_weather(value)
            if success:
                weather_temp = temp
                await update.message.reply_text(f"🌤️ Погода в {value}: {temp}°C")
            else:
                await update.message.reply_text(f"⚠️ Не удалось получить погоду для {value}, использую 20°C")

        elif param == 'вес':
            new_weight = float(value)
            if not (30 <= new_weight <= 200):
                await update.message.reply_text("❌ Вес должен быть от 30 до 200 кг")
                return
            profile.weight = new_weight

        elif param == 'рост':
            new_height = float(value)
            if not (100 <= new_height <= 250):
                await update.message.reply_text("❌ Рост должен быть от 100 до 250 см")
                return
            profile.height = new_height

        elif param == 'возраст':
            new_age = int(value)
            if not (10 <= new_age <= 120):
                await update.message.reply_text("❌ Возраст должен быть от 10 до 120 лет")
                return
            profile.age = new_age

        elif param == 'активность':
            if value.lower() not in ['sedentary', 'light', 'moderate', 'active', 'very_active']:
                await update.message.reply_text(
                    "❌ Неверный уровень активности. Допустимые значения:\n"
                    "• sedentary (сидячий)\n"
                    "• light (легкий)\n"
                    "• moderate (умеренный)\n"
                    "• active (активный)\n"
                    "• very_active (очень активный)"
                )
                return
            profile.activity_level = value.lower()

        elif param == 'цель':
            if value.lower() not in ['lose', 'maintain', 'gain']:
                await update.message.reply_text(
                    "❌ Неверная цель. Допустимые значения:\n"
                    "• lose (похудеть)\n"
                    "• maintain (поддерживать)\n"
                    "• gain (набрать массу)"
                )
                return
            profile.goal = value.lower()

        else:
            await update.message.reply_text(f"❌ Неизвестный параметр: {param}")
            return

        if param == 'город':
            Calculator.calculate_goals(profile, weather_temp)
        else:
            temp, success = await get_weather(profile.city)
            Calculator.calculate_goals(profile, temp if success else 20)

        def format_param(param_name, param_value):
            if param_name == 'активность':
                activity_names = {
                    'sedentary': 'Сидячий',
                    'light': 'Легкая',
                    'moderate': 'Умеренная',
                    'active': 'Активная',
                    'very_active': 'Очень активная'
                }
                return activity_names.get(param_value, param_value)
            elif param_name == 'цель':
                goal_names = {
                    'lose': 'Похудеть',
                    'maintain': 'Поддерживать вес',
                    'gain': 'Набрать массу'
                }
                return goal_names.get(param_value, param_value)
            elif param_name == 'город':
                return param_value
            else:
                return str(param_value)

        formatted_old = format_param(param, old_value) if old_value else "не установлено"
        formatted_new = format_param(param, value)

        response = f"""
        ✅ *Параметр успешно изменен!*

        • **{param.capitalize()}:** {formatted_old} → {formatted_new}

        📊 *Обновленные цели:*
        • 💧 Вода: {profile.water_goal} мл
        • 🔥 Калории: {profile.calorie_goal} ккал
        • 🥚 Белки: {profile.protein_goal} г
        • 🥑 Жиры: {profile.fat_goal} г
        • 🍚 Углеводы: {profile.carbs_goal} г

        Для просмотра полного профиля используйте /profile
        """

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

    except ValueError:
        await update.message.reply_text(f"❌ Неверный формат значения для параметра '{param}'")
    except Exception as e:
        logger.error(f"Ошибка при редактировании профиля: {e}")
        await update.message.reply_text("❌ Произошла ошибка при редактировании профиля")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """
    🤖 *Fitness Tracker Pro Bot - Полная справка*

    *👤 Профиль и настройки:*
    /profile - Настроить или посмотреть профиль
    /editprofile - Изменить параметры профиля
    /start - Запустить бота и посмотреть возможности

    *📊 Отслеживание:*
    /logwater <мл> - Записать потребление воды
    /logfood <продукт> [граммы] - Записать прием пищи (100г по умолчанию)
    /logworkout <тип> <минуты> - Записать тренировку

    *📈 Аналитика и визуализация:*
    /progress [дни] - Графики прогресса (по умолчанию 7 дней)
    /macros - Диаграмма макронутриентов
    /stats - Детальная статистика за день

    *🎯 Рекомендации и советы:*
    /recommend - Персонализированные рекомендации
    /tips - Ежедневные советы

    *🔄 Управление данными:*
    /reset - Сбросить все данные

    *Примеры использования:*
    • /logwater 500
    • /logfood куриная грудка 150
    • /logworkout бег 45
    • /progress 14
    • /editprofile город Самара

    *Типы тренировок:*
    бег, ходьба, велосипед, плавание, йога, пилатес, силовая, кардио, танцы, бокс, кроссфит

    *✨ Премиум-функции:*
    • Умный поиск калорийности через USDA API
    • Графики прогресса за любой период
    • Анализ макронутриентов (белки, жиры, углеводы)
    • Персонализированные рекомендации
    • Ежедневные советы на основе ваших целей

    *ℹ️ Примечание:*
    **Данные НЕ сбрасываются автоматически в полночь.**
    Для сброса используйте команду /reset
    Для персонализации требуется заполнить профиль.
    """

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    command_text = update.message.text if update.message else "/reset"
    log_command(user.id, user.username or "no username", command_text)

    user_id = update.effective_user.id

    keyboard = [
        [
            InlineKeyboardButton("Да, сбросить все", callback_data="reset_confirm"),
            InlineKeyboardButton("Отмена", callback_data="reset_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "⚠️ *Вы уверены, что хотите сбросить все данные?*\n\n"
        "Это действие удалит:\n"
        "• Ваш профиль\n"
        "• Все логи питания и тренировок\n\n"
        "Данные НЕ сбрасываются автоматически!\n"
        "Это действие необратимо!",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )


async def handle_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    gender = 'male' if query.data == 'gender_male' else 'female'

    if user_id not in profile_temp_data:
        profile_temp_data[user_id] = {}
    profile_temp_data[user_id]['gender'] = gender
    profile_temp_data[user_id]['timestamp'] = datetime.now()

    context.user_data['gender'] = gender
    logger.info(f"Сохранен пол для пользователя {user_id}: {gender}")

    keyboard = [
        [
            InlineKeyboardButton("Сидячий образ жизни", callback_data="activity_sedentary"),
            InlineKeyboardButton("Легкая активность", callback_data="activity_light")
        ],
        [
            InlineKeyboardButton("Умеренная активность", callback_data="activity_moderate"),
            InlineKeyboardButton("Высокая активность", callback_data="activity_active")
        ],
        [InlineKeyboardButton("Очень высокая активность", callback_data="activity_very_active")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🏃‍♂️ Выберите уровень вашей активности:",
        reply_markup=reply_markup
    )

async def handle_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    activity_map = {
        'activity_sedentary': 'sedentary',
        'activity_light': 'light',
        'activity_moderate': 'moderate',
        'activity_active': 'active',
        'activity_very_active': 'very_active'
    }

    activity_level = activity_map[query.data]

    if user_id not in profile_temp_data:
        profile_temp_data[user_id] = {}
    profile_temp_data[user_id]['activity_level'] = activity_level
    profile_temp_data[user_id]['timestamp'] = datetime.now()

    context.user_data['activity_level'] = activity_level
    logger.info(f"Сохранена активность для пользователя {user_id}: {activity_level}")

    keyboard = [
        [InlineKeyboardButton("Похудеть ⬇️", callback_data="goal_lose")],
        [InlineKeyboardButton("Поддерживать вес ⏸️", callback_data="goal_maintain")],
        [InlineKeyboardButton("Набрать массу ⬆️", callback_data="goal_gain")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🎯 Выберите вашу цель:",
        reply_markup=reply_markup
    )

async def handle_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    goal_map = {
        'goal_lose': 'lose',
        'goal_maintain': 'maintain',
        'goal_gain': 'gain'
    }

    goal = goal_map[query.data]

    if user_id not in profile_temp_data:
        profile_temp_data[user_id] = {}
    profile_temp_data[user_id]['goal'] = goal
    profile_temp_data[user_id]['timestamp'] = datetime.now()

    context.user_data['goal'] = goal
    logger.info(f"Сохранена цель для пользователя {user_id}: {goal}")

    await query.edit_message_text(
        "📝 *Введите данные через пробел:*\n"
        "`<вес в кг> <рост в см> <возраст> <город>`\n\n"
        "*Пример:* `70 175 25 Самара`\n\n"
        "ℹ️ *Важно:* Город нужен для учета погоды при расчете нормы воды.",
        parse_mode=ParseMode.MARKDOWN
    )

async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    if query.data == "reset_confirm":
        if user_id in user_profiles:
            del user_profiles[user_id]
        if user_id in user_logs:
            del user_logs[user_id]

        await query.edit_message_text(
            "✅ Все данные сброшены!\n\n"
            "Используйте /profile для создания нового профиля.",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await query.edit_message_text(
            "❌ Сброс отменен.",
            parse_mode=ParseMode.MARKDOWN
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")

    try:
        await update.message.reply_text(
            "😕 Произошла ошибка. Пожалуйста, попробуйте еще раз.\n"
            "Если проблема повторяется, используйте /help для справки."
        )
    except:
        pass

async def cleanup_old_temp_data():
    import asyncio
    from datetime import datetime, timedelta

    while True:
        try:
            current_time = datetime.now()
            users_to_remove = []

            for user_id, data in profile_temp_data.items():
                if 'timestamp' in data:
                    if current_time - data['timestamp'] > timedelta(minutes=5):
                        users_to_remove.append(user_id)

            for user_id in users_to_remove:
                del profile_temp_data[user_id]
                logger.info(f"Очищены старые данные профиля для пользователя {user_id}")

            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Ошибка при очистке временных данных: {e}")
            await asyncio.sleep(60)

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'ВАШ_ТОКЕН_ТЕЛЕГРАМ_БОТА':
        exit(1)

    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("profile", profile_command))
        application.add_handler(CommandHandler("editprofile", editprofile_command))
        application.add_handler(CommandHandler("logwater", logwater_command))
        application.add_handler(CommandHandler("logfood", logfood_command))
        application.add_handler(CommandHandler("logworkout", logworkout_command))
        application.add_handler(CommandHandler("progress", progress_command))
        application.add_handler(CommandHandler("macros", macros_command))
        application.add_handler(CommandHandler("recommend", recommend_command))
        application.add_handler(CommandHandler("tips", tips_command))
        application.add_handler(CommandHandler("stats", stats_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("reset", reset_command))

        application.add_handler(CallbackQueryHandler(handle_gender, pattern="^gender_"))
        application.add_handler(CallbackQueryHandler(handle_activity, pattern="^activity_"))
        application.add_handler(CallbackQueryHandler(handle_goal, pattern="^goal_"))
        application.add_handler(CallbackQueryHandler(handle_reset, pattern="^reset_"))

        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_profile_data))

        application.add_error_handler(error_handler)

        import asyncio
        loop = asyncio.get_event_loop()
        loop.create_task(cleanup_old_temp_data())

        application.run_polling(allowed_updates=Update.ALL_TYPES)

    except Exception as e:
        print(f"❌ Ошибка запуска бота: {e}")
        print("ℹ️ Проверьте токен бота и соединение с интернетом")

if __name__ == '__main__':
    main()
