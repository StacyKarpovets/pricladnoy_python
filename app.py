import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import asyncio
import aiohttp
import time
from typing import List, Dict
from datetime import datetime

API_KEY = ""

seasonal_temperatures = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
}

month_to_season = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn"
}

ALL_CITIES = list(seasonal_temperatures.keys())

@st.cache_data
def generate_realistic_temperature_data(cities: List[str], num_years: int = 6):
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []

    for city in cities:
        city_temps = []
        
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            
            if season == "winter":
                scale = 6
            elif season == "summer":
                scale = 4
            else:
                scale = 5
            
            anomaly_factor = 0
            if city == "Beijing" and date.year >= 2021:
                anomaly_factor += np.random.uniform(3, 8)
            if city == "Moscow" and date.year >= 2020:
                anomaly_factor += np.random.uniform(2, 6)
            
            temperature = np.random.normal(loc=mean_temp + anomaly_factor, scale=scale)
            
            city_temps.append({
                "city": city,
                "timestamp": date,
                "temperature": round(temperature, 1),
                "season": season,
                "year": date.year,
                "month": date.month
            })
        
        data.extend(city_temps)
        
        if len(city_temps) > 1:
            for i in range(1, len(data) - len(city_temps), len(city_temps)):
                prev_temp = data[i-1]["temperature"]
                current_temp = data[i]["temperature"]
                if abs(current_temp - prev_temp) > 8:
                    data[i]["temperature"] = prev_temp + np.random.normal(0, 3)
    
    df = pd.DataFrame(data)
    
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    
    return df

def analyze_city_data(city_data):
    city_data = city_data.copy().sort_values('timestamp')
    
    city_data['rolling_mean_30d'] = city_data['temperature'].rolling(window=30, center=True, min_periods=1).mean()
    city_data['rolling_std_30d'] = city_data['temperature'].rolling(window=30, center=True, min_periods=1).std()
    city_data['rolling_mean_7d'] = city_data['temperature'].rolling(window=7, center=True, min_periods=1).mean()
    
    city_data['is_anomaly'] = (
        (city_data['temperature'] > city_data['rolling_mean_30d'] + 2 * city_data['rolling_std_30d']) |
        (city_data['temperature'] < city_data['rolling_mean_30d'] - 2 * city_data['rolling_std_30d'])
    )
    
    yearly_stats = city_data.groupby('year').agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(1)
    yearly_stats.columns = ['mean', 'std', 'min', 'max', 'count']
    
    seasonal_stats = city_data.groupby('season').agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(1)
    seasonal_stats.columns = ['mean', 'std', 'min', 'max', 'count']
    
    city_data['days_since_start'] = (city_data['timestamp'] - city_data['timestamp'].min()).dt.days
    if len(city_data) > 1:
        z = np.polyfit(city_data['days_since_start'], city_data['temperature'], 1)
        city_data['trend_line'] = np.poly1d(z)(city_data['days_since_start'])
        trend_slope = z[0] * 365
    else:
        city_data['trend_line'] = city_data['temperature']
        trend_slope = 0
    
    overall_stats = {
        'mean': round(city_data['temperature'].mean(), 1),
        'std': round(city_data['temperature'].std(), 1),
        'min': round(city_data['temperature'].min(), 1),
        'max': round(city_data['temperature'].max(), 1),
        'median': round(city_data['temperature'].median(), 1),
        'q1': round(city_data['temperature'].quantile(0.25), 1),
        'q3': round(city_data['temperature'].quantile(0.75), 1),
        'total_days': len(city_data),
        'anomaly_days': city_data['is_anomaly'].sum(),
        'anomaly_percent': round(city_data['is_anomaly'].sum() / len(city_data) * 100, 1),
        'trend_per_year': round(trend_slope, 2)
    }
    
    return {
        'data': city_data,
        'seasonal_stats': seasonal_stats,
        'yearly_stats': yearly_stats,
        'overall_stats': overall_stats
    }

def get_current_weather_sync(api_key: str, city: str) -> Dict:
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'city': data['name'],
                'temperature': round(data['main']['temp'], 1),
                'feels_like': round(data['main']['feels_like'], 1),
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'cloudiness': data['clouds']['all'],
                'visibility': data.get('visibility', 0),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        elif response.status_code == 401:
            error_data = response.json()
            error_msg = error_data.get('message', 'Invalid API key')
            return {
                'success': False, 
                'error': f'Неверный API ключ: {error_msg}. Получите ключ на openweathermap.org',
                'api_error': error_data
            }
        elif response.status_code == 404:
            return {'success': False, 'error': f'Город {city} не найден'}
        else:
            return {'success': False, 'error': f'Ошибка API: {response.status_code}'}
            
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Таймаут запроса'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def fetch_weather(session: aiohttp.ClientSession, api_key: str, city: str) -> Dict:
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        async with session.get(url, params=params, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'success': True,
                    'city': data['name'],
                    'temperature': round(data['main']['temp'], 1),
                    'feels_like': round(data['main']['feels_like'], 1),
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed']
                }
            elif response.status == 401:
                error_data = await response.json()
                return {
                    'success': False, 
                    'error': 'Invalid API key.', 
                    'city': city,
                    'api_error': error_data
                }
            else:
                return {'success': False, 'error': f'API Error: {response.status}', 'city': city}
                
    except Exception as e:
        return {'success': False, 'error': str(e), 'city': city}

async def get_multiple_weather_async(api_key: str, cities: List[str]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_weather(session, api_key, city) for city in cities]
        results = await asyncio.gather(*tasks)
        return results

def create_temperature_timeseries(city_data, city_name):
    fig = go.Figure()
  
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['temperature'],
        mode='lines',
        name='Температура',
        line=dict(color='rgba(100, 149, 237, 0.7)', width=1),
        hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Температура: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=city_data['timestamp'],
        y=city_data['rolling_mean_30d'],
        mode='lines',
        name='Скользящее среднее (30 дней)',
        line=dict(color='blue', width=2),
        hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Среднее: %{y:.1f}°C<extra></extra>'
    ))
    
    if 'trend_line' in city_data.columns:
        fig.add_trace(go.Scatter(
            x=city_data['timestamp'],
            y=city_data['trend_line'],
            mode='lines',
            name='Долгосрочный тренд',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Тренд: %{y:.1f}°C<extra></extra>'
        ))
    
    anomalies = city_data[city_data['is_anomaly']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            name='Аномалии (±2σ)',
            marker=dict(color='red', size=8, symbol='circle-open', line=dict(width=2)),
            hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Температура: %{y:.1f}°C<br>Аномалия<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Температура в {city_name} ({city_data["timestamp"].min().year}-{city_data["timestamp"].max().year})',
        xaxis_title='Дата',
        yaxis_title='Температура (°C)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_seasonal_boxplot(city_data, city_name, seasonal_stats):
    fig = px.box(
        city_data, 
        x='season', 
        y='temperature',
        color='season',
        points=False,
        category_orders={'season': ['winter', 'spring', 'summer', 'autumn']}
    )
    
    for season in ['winter', 'spring', 'summer', 'autumn']:
        if season in seasonal_stats.index:
            season_mean = seasonal_stats.loc[season, 'mean']
            fig.add_hline(
                y=season_mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Среднее: {season_mean}°C",
                annotation_position="top right"
            )
    
    fig.update_layout(
        title=f'Распределение температур по сезонам в {city_name}',
        xaxis_title='Сезон',
        yaxis_title='Температура (°C)',
        height=450,
        showlegend=False
    )
    
    return fig

def create_yearly_trend_chart(yearly_stats, city_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_stats.index,
        y=yearly_stats['mean'],
        mode='lines+markers',
        name='Средняя температура',
        line=dict(color='blue', width=3),
        marker=dict(size=10),
        hovertemplate='<b>%{x}</b><br>Среднее: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(yearly_stats.index) + list(yearly_stats.index)[::-1],
        y=list(yearly_stats['mean'] + yearly_stats['std']) + list(yearly_stats['mean'] - yearly_stats['std'])[::-1],
        fill='toself',
        fillcolor='rgba(100, 149, 237, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 стандартное отклонение',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Изменение средней температуры по годам в {city_name}',
        xaxis_title='Год',
        yaxis_title='Температура (°C)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Анализ температурных данных",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stMetric label {
        font-weight: bold !important;
    }
    .city-header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .api-key-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌡️ Анализ температурных данных")
    
    with st.sidebar:
        selected_city = st.selectbox(
            "Выберите город для анализа:",
            ALL_CITIES,
            index=6,
            help="Выберите город для детального анализа температурных данных"
        )
        
        st.markdown("---")
        
        st.markdown('<div class="api-key-form">', unsafe_allow_html=True)
        st.subheader("🔑 OpenWeatherMap API")
        
        if 'api_key_valid' not in st.session_state:
            st.session_state.api_key_valid = False
        if 'api_key_error' not in st.session_state:
            st.session_state.api_key_error = None
        
        api_key_input = st.text_input(
            "Введите ваш API ключ:",
            value="",
            type="password",
            placeholder="Введите ключ OpenWeatherMap..."
        )
        
        col_check1, col_check2 = st.columns([2, 1])
        with col_check1:
            check_key = st.button("Проверить ключ", use_container_width=True)
        
        with col_check2:
            clear_key = st.button("Очистить", use_container_width=True, type="secondary")
        
        if clear_key:
            st.session_state.api_key_valid = False
            st.session_state.api_key_error = None
            st.rerun()
        
        if check_key:
            if api_key_input:
                with st.spinner("Проверка API ключа..."):
                    test_result = get_current_weather_sync(api_key_input, "London")
                    
                    if test_result['success']:
                        st.session_state.api_key_valid = True
                        st.session_state.api_key_error = None
                        st.success("API ключ действителен!")
                    else:
                        st.session_state.api_key_valid = False
                        st.session_state.api_key_error = test_result
                        
                        if test_result.get('api_error', {}).get('cod') == 401:
                            st.error(f"{test_result['error']}")
                        else:
                            st.warning(f"{test_result.get('error', 'Неизвестная ошибка')}")
            else:
                st.warning("Введите API ключ для проверки")
        
        if st.session_state.api_key_valid:
            st.success("API ключ настроен")
        elif st.session_state.api_key_error:
            error = st.session_state.api_key_error
            if error.get('api_error', {}).get('cod') == 401:
                st.error("Неверный API ключ")
            else:
                st.warning("Проблема с API ключом")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Параметры анализа")
        
        method = st.radio(
            "Выберите метод получения погоды:",
            ["Синхронный", "Асинхронный"],
            index=0
        )
        
        years_to_show = st.multiselect(
            "Годы для анализа:",
            options=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            default=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            help="Выберите годы для включения в анализ")
    
    with st.spinner("Генерация реалистичных температурных данных..."):
        data = generate_realistic_temperature_data(ALL_CITIES)
    
    if years_to_show:
        data = data[data['year'].isin(years_to_show)]
    
    city_data_filtered = data[data['city'] == selected_city]
    
    analysis = analyze_city_data(city_data_filtered)
    city_data = analysis['data']
    overall_stats = analysis['overall_stats']
    seasonal_stats = analysis['seasonal_stats']
    yearly_stats = analysis['yearly_stats']
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Статистика", 
        "📈 Анализ трендов", 
        "🌡️ Текущая погода",
        "⚡ Производительность"
    ])
    
    with tab1:
        st.markdown(f'<div class="city-header">📊 Статистика {selected_city}</div>', unsafe_allow_html=True)
        
        st.subheader("📈 Основные показатели")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_temp = overall_stats['trend_per_year']
            delta_color = "inverse" if delta_temp < 0 else "normal"
            st.metric(
                "Средняя температура", 
                f"{overall_stats['mean']}°C",
                f"{delta_temp:+.2f}°C/год",
                delta_color=delta_color
            )
        
        with col2:
            st.metric("Стандартное отклонение", f"{overall_stats['std']}°C")
        
        with col3:
            anomaly_percent = overall_stats['anomaly_percent']
            anomaly_color = "normal" if anomaly_percent < 5 else "off"
            st.metric(
                "Аномальных дней", 
                f"{overall_stats['anomaly_days']}",
                f"{anomaly_percent}%",
                delta_color=anomaly_color
            )
      
        with col4:
            temp_range = overall_stats['max'] - overall_stats['min']
            st.metric("📏 Диапазон температур", f"{temp_range:.1f}°C")
        
        fig1 = create_temperature_timeseries(city_data, selected_city)
        st.plotly_chart(fig1, use_container_width=True)
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            fig2 = create_seasonal_boxplot(city_data, selected_city, seasonal_stats)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_right:
            st.subheader("Статистика по сезонам")
            display_stats = seasonal_stats.copy()
            display_stats = display_stats[['mean', 'std', 'min', 'max']]
            display_stats.columns = ['Средняя', 'Стд. откл.', 'Минимум', 'Максимум']
          
            def format_temp(val):
                return f"{val:.1f}°C"
            
            for col in display_stats.columns:
                display_stats[col] = display_stats[col].apply(format_temp)
            
            st.dataframe(
                display_stats,
                use_container_width=True,
                height=350
            )
            
            st.markdown("**Климатическая норма:**")
            climate_norms = seasonal_temperatures[selected_city]
            for season, temp in climate_norms.items():
                st.write(f"{season.capitalize()}: {temp}°C")
    
    with tab2:
        st.header(f"📈 Анализ температурных трендов в {selected_city}")
        
        st.subheader("Годовая динамика")
        fig_yearly = create_yearly_trend_chart(yearly_stats, selected_city)
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        st.subheader("Сезонность")
        
        pivot_data = city_data.pivot_table(
            index='year',
            columns='season',
            values='temperature',
            aggfunc='mean'
        ).reindex(columns=['winter', 'spring', 'summer', 'autumn'])
        
        fig_seasonal = go.Figure()
        
        for season in ['winter', 'spring', 'summer', 'autumn']:
            if season in pivot_data.columns:
                fig_seasonal.add_trace(go.Scatter(
                    x=pivot_data.index,
                    y=pivot_data[season],
                    mode='lines+markers',
                    name=season.capitalize(),
                    hovertemplate=f'{season.capitalize()}: %{{y:.1f}}°C<extra></extra>'
                ))
        
        fig_seasonal.update_layout(
            title=f'Средняя температура по сезонам в {selected_city}',
            xaxis_title='Год',
            yaxis_title='Температура (°C)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.subheader("Распределение температур")
            
            fig_hist = px.histogram(
                city_data,
                x='temperature',
                nbins=50,
                title=f'Распределение температур в {selected_city}',
                labels={'temperature': 'Температура (°C)'}
            )
          
            fig_hist.add_vline(
                x=overall_stats['mean'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Среднее: {overall_stats['mean']}°C"
            )
            
            fig_hist.add_vline(
                x=overall_stats['median'],
                line_dash="dot",
                line_color="green",
                annotation_text=f"Медиана: {overall_stats['median']}°C"
            )
            
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_stat2:
            st.subheader("Корреляционный анализ")
            
            monthly_data = city_data.copy()
            monthly_data['month_name'] = monthly_data['timestamp'].dt.month_name()
            
            heatmap_data = monthly_data.pivot_table(
                index='year',
                columns='month',
                values='temperature',
                aggfunc='mean'
            )
            
            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Месяц", y="Год", color="Температура (°C)"),
                x=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                   'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'],
                title=f'Температурная карта по месяцам и годам'
            )
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.header("🌤️ Текущая погода")
        
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            if st.button("Получить текущую погоду", type="primary", use_container_width=True):
                with st.spinner(f"Запрос данных для {selected_city}..."):
                    start_time = time.time()
                    
                    if method == "Синхронный":
                        weather_data = get_current_weather_sync(api_key_input, selected_city)
                    else:
                        weather_data = get_current_weather_sync(api_key_input, selected_city)
                    
                    request_time = time.time() - start_time
                    
                    if weather_data['success']:
                        st.session_state['weather_data'] = weather_data
                        st.session_state['request_time'] = request_time
                        st.success(f"Данные получены за {request_time:.2f} секунд")
                    else:
                        st.error(f"Ошибка: {weather_data['error']}")
                        
                        if weather_data.get('api_error', {}).get('cod') == 401:
                            with st.expander("Ошибка с API"):
                                st.json(weather_data['api_error'])
        
        with col_btn2:
            if st.button("Очистить данные", type="secondary", use_container_width=True):
                if 'weather_data' in st.session_state:
                    del st.session_state['weather_data']
                st.rerun()
        
        if 'weather_data' in st.session_state and st.session_state['weather_data']['success']:
            weather = st.session_state['weather_data']
            
            st.markdown("---")
            st.subheader(f"Текущая погода в {weather['city']}")
            
            cols_weather = st.columns(4)
            with cols_weather[0]:
                st.metric("🌡️ Температура", f"{weather['temperature']}°C")
            with cols_weather[1]:
                st.metric("💨 Ощущается как", f"{weather['feels_like']}°C")
            with cols_weather[2]:
                st.metric("💧 Влажность", f"{weather['humidity']}%")
            with cols_weather[3]:
                st.metric("🔽 Давление", f"{weather['pressure']} hPa")
            
            col_desc, col_sun = st.columns(2)
            with col_desc:
                st.info(f"**🌤️ Погодные условия:** {weather['description'].capitalize()}")
            with col_sun:
                st.info(f"**🌅 Восход:** {weather['sunrise']} | **🌇 Закат:** {weather['sunset']}")
            
            st.markdown("---")
            st.subheader("Сравнение с историческими данными")
            
            current_temp = weather['temperature']
            hist_mean = overall_stats['mean']
            hist_std = overall_stats['std']
            
            deviation = current_temp - hist_mean
            z_score = deviation / hist_std if hist_std > 0 else 0
            
            if abs(z_score) <= 2:
                status = "✅ **Температура в пределах нормы**"
                color = "green"
                icon = "✅"
            elif abs(z_score) <= 3:
                status = "⚠️ **Температура нестандартная**"
                color = "orange"
                icon = "⚠️"
            else:
                status = "🚨 **Температура аномальная**"
                color = "red"
                icon = "🚨"
            
            st.markdown(f"""
            <div style="background-color:{color}20; padding:15px; border-radius:10px; border-left:5px solid {color};">
                <h4>{icon} {status}</h4>
                <p><b>Текущая температура:</b> {current_temp}°C</p>
                <p><b>Историческое среднее ({years_to_show[0]}-{years_to_show[-1]}):</b> {hist_mean}°C</p>
                <p><b>Отклонение:</b> <span style="color:{'red' if deviation > 0 else 'blue'}">{deviation:+.1f}°C</span></p>
                <p><b>Z-оценка:</b> {z_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Indicator(
                mode="number+delta",
                value=current_temp,
                delta={'reference': hist_mean, 'relative': False, 'valueformat': '.1f'},
                title={'text': "Текущая температура"},
                domain={'row': 0, 'column': 0}
            ))
            
            fig_comparison.add_trace(go.Indicator(
                mode="number",
                value=hist_mean,
                title={'text': "Историческое среднее"},
                domain={'row': 0, 'column': 1}
            ))
            
            fig_comparison.add_trace(go.Indicator(
                mode="number",
                value=abs(z_score),
                title={'text': "Z-оценка"},
                domain={'row': 0, 'column': 2}
            ))
            
            fig_comparison.update_layout(
                grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
                height=200
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab4:
        st.header("Сравнение асинхронных и синхронных запросов")
        
        st.markdown("""
        ### Сравнение синхронных и асинхронных запросов к API
        """)
        
        if not st.session_state.api_key_valid:
            st.warning("Требуется действительный API ключ")
        elif st.button("Запустить тест производительности", type="primary", use_container_width=True):
            test_cities = ["Berlin", "Paris", "London", "Tokyo", "Moscow", "New York"]
            
            st.info(f"Тестирование для {len(test_cities)} городов: {', '.join(test_cities)}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⏳ Выполнение синхронных запросов...")
            sync_results = []
            sync_times = []
            start_time = time.time()
            
            for i, city in enumerate(test_cities):
                city_start = time.time()
                result = get_current_weather_sync(api_key_input, city)
                city_time = time.time() - city_start
                
                sync_results.append(result)
                sync_times.append(city_time)
                progress_bar.progress((i + 1) / (len(test_cities) * 2))
                time.sleep(0.1)
            
            sync_total_time = time.time() - start_time
            
            status_text.text("Выполнение асинхронных запросов...")
            start_time = time.time()
            
            async def run_async_test():
                return await get_multiple_weather_async(api_key_input, test_cities)
            
            async_results = asyncio.run(run_async_test())
            
            for i in range(len(test_cities)):
                progress_bar.progress((len(test_cities) + i + 1) / (len(test_cities) * 2))
            
            async_total_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text("Тестирование завершено!")
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.metric("Синхронные запросы", 
                         f"{sync_total_time:.2f} сек",
                         f"{sync_total_time/len(test_cities):.2f} сек/город",
                         delta_color="normal")
                
                with st.expander("Детали синхронных запросов"):
                    sync_df = pd.DataFrame({
                        'Город': test_cities,
                        'Время (сек)': [f"{t:.2f}" for t in sync_times],
                        'Статус': ['✅' if r['success'] else '❌' for r in sync_results],
                        'Температура': [f"{r['temperature']}°C" if r['success'] else 'Ошибка' for r in sync_results]
                    })
                    st.dataframe(sync_df, use_container_width=True, hide_index=True)
            
            with col_perf2:
                st.metric("Асинхронные запросы", 
                         f"{async_total_time:.2f} сек",
                         f"{async_total_time/len(test_cities):.2f} сек/город",
                         delta_color="normal")
                
                with st.expander("Детали асинхронных запросов"):
                    async_df = pd.DataFrame({
                        'Город': test_cities,
                        'Статус': ['✅' if r['success'] else '❌' for r in async_results],
                        'Температура': [f"{r['temperature']}°C" if r['success'] else r.get('error', 'Ошибка') for r in async_results]
                    })
                    st.dataframe(async_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("Сравнительный анализ производительности")
            
            if sync_total_time > 0 and async_total_time > 0:
                speedup = sync_total_time / async_total_time
                
                perf_data = pd.DataFrame({
                    'Метод': ['Синхронный', 'Асинхронный'],
                    'Общее время (сек)': [sync_total_time, async_total_time],
                    'Время на город (сек)': [sync_total_time/len(test_cities), async_total_time/len(test_cities)]
                })
                
                fig_perf = px.bar(
                    perf_data,
                    x='Метод',
                    y='Общее время (сек)',
                    color='Метод',
                    title='Сравнение времени выполнения запросов',
                    text='Общее время (сек)'
                )
                
                fig_perf.update_traces(texttemplate='%{text:.2f} сек', textposition='outside')
                fig_perf.update_layout(height=400, showlegend=False)
                
                st.plotly_chart(fig_perf, use_container_width=True)
              
                if speedup > 1.2:
                    st.success(f"**Ускорение в {speedup:.1f} раза при использовании асинхронных запросов!**")
                    st.markdown("""
                    **Выводы:**
                    - Асинхронные запросы значительно эффективнее для одновременного получения данных по нескольким городам
                    - Экономия времени растет с увеличением количества запрашиваемых городов
                    - Рекомендуется использовать асинхронный метод для мониторинга 3+ городов
                    """)
                else:
                    st.warning("**Разница в производительности незначительна**")
                    st.markdown("""
                    **Выводы:**
                    - Для небольшого количества городов (1-3) разница между методами минимальна
                    - Синхронный метод проще в реализации и отладке
                    - Выбор метода зависит от конкретных требований приложения
                    """)

if __name__ == "__main__":
    main()
