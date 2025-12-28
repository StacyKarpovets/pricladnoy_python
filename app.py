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
def generate_realistic_temperature_data(cities: List[str], num_years: int = 15):
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
    
    df = pd.DataFrame(data)
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    
    return df

def analyze_city_data(city_data):
    city_data = city_data.copy().sort_values('timestamp')
    
    city_data['rolling_mean_30d'] = city_data['temperature'].rolling(window=30, center=True, min_periods=1).mean()
    city_data['rolling_std_30d'] = city_data['temperature'].rolling(window=30, center=True, min_periods=1).std()
    
    city_data['is_anomaly'] = (
        (city_data['temperature'] > city_data['rolling_mean_30d'] + 2 * city_data['rolling_std_30d']) |
        (city_data['temperature'] < city_data['rolling_mean_30d'] - 2 * city_data['rolling_std_30d'])
    )
    
    yearly_stats = city_data.groupby('year').agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(1)
    yearly_stats.columns = ['mean', 'std', 'min', 'max', 'count']
    
    seasonal_stats = city_data.groupby('season', as_index=False).agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    })
    
    seasonal_stats.columns = ['season', 'mean', 'std', 'min', 'max', 'count']
    
    for col in ['mean', 'std', 'min', 'max']:
        seasonal_stats[col] = seasonal_stats[col].round(1)
    
    seasonal_stats_indexed = seasonal_stats.set_index('season')
    
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
        'anomaly_days': city_data['is_anomaly'].sum(),
        'anomaly_percent': round(city_data['is_anomaly'].sum() / len(city_data) * 100, 1),
        'trend_per_year': round(trend_slope, 2)
    }
    
    return {
        'data': city_data,
        'seasonal_stats': seasonal_stats_indexed,
        'yearly_stats': yearly_stats,
        'overall_stats': overall_stats,
    }
    
def get_current_weather_sync(api_key: str, city: str) -> Dict:
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {'q': city, 'appid': api_key, 'units': 'metric', 'lang': 'ru'}
        
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
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
            }
        elif response.status_code == 401:
            return {'success': False, 'error': 'Неверный API ключ'}
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
        params = {'q': city, 'appid': api_key, 'units': 'metric', 'lang': 'ru'}
        
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
                    'wind_speed': data['wind']['speed'],
                    'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                    'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
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
        title=f'Температура в {city_name}',
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
    
    seasons_in_data = seasonal_stats.index.tolist()
    
    for season in ['winter', 'spring', 'summer', 'autumn']:
        if season in seasons_in_data:
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
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .city-header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌡️ Анализ температурных данных")
    
    with st.sidebar:
        st.header("Настройки")
        
        selected_city = st.selectbox("Выберите город:", ALL_CITIES, index=6)
        
        st.markdown("---")
        
        if 'api_key_valid' not in st.session_state:
            st.session_state.api_key_valid = False
        
        api_key_input = st.text_input("API ключ OpenWeatherMap:", type="password")
        
        if st.button("Проверить ключ"):
            if api_key_input:
                with st.spinner("Проверка..."):
                    test_result = get_current_weather_sync(api_key_input, "London")
                    if test_result['success']:
                        st.session_state.api_key_valid = True
                        st.success("API ключ действителен!")
                    else:
                        st.session_state.api_key_valid = False
                        st.error(test_result['error'])
            else:
                st.warning("Введите API ключ")
        
        st.markdown("---")
        
        method = st.radio("Метод запроса:", ["Синхронный", "Асинхронный"], index=0)
        
        years_to_show = st.multiselect(
            "Годы для анализа:",
            options=list(range(2010, 2025)),
            default=list(range(2010, 2025))
        )
    
    with st.spinner("Генерация данных..."):
        data = generate_realistic_temperature_data(ALL_CITIES)
    
    if years_to_show:
        data = data[data['year'].isin(years_to_show)]
    
    city_data_filtered = data[data['city'] == selected_city]
    analysis = analyze_city_data(city_data_filtered)
    city_data = analysis['data']
    overall_stats = analysis['overall_stats']
    seasonal_stats = analysis['seasonal_stats']
    yearly_stats = analysis['yearly_stats']
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Статистика", "📈 Динамика", "🌡️ Погода", "⚡ Сравнение скорости"])
    
    with tab1:
        st.markdown(f'<div class="city-header">📊 Статистика {selected_city}</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_temp = overall_stats['trend_per_year']
            delta_color = "inverse" if delta_temp < 0 else "normal"
            st.metric("Средняя температура", f"{overall_stats['mean']}°C", f"{delta_temp:+.2f}°C/год", delta_color=delta_color)
        
        with col2:
            st.metric("Стандартное отклонение", f"{overall_stats['std']}°C")
        
        with col3:
            anomaly_percent = overall_stats['anomaly_percent']
            st.metric("Аномальных дней", f"{overall_stats['anomaly_days']}", f"{anomaly_percent}%")
        
        with col4:
            temp_range = overall_stats['max'] - overall_stats['min']
            st.metric("Диапазон температур", f"{temp_range:.1f}°C")
        
        fig1 = create_temperature_timeseries(city_data, selected_city)
        st.plotly_chart(fig1, use_container_width=True)
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig2 = create_seasonal_boxplot(city_data, selected_city, seasonal_stats)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_right:
            st.subheader("Статистика по сезонам")
            
            seasons_data = []
            for season in ['winter', 'spring', 'summer', 'autumn']:
                if season in seasonal_stats.index:
                    season_data = seasonal_stats.loc[season]
                    seasons_data.append({
                        'Сезон': season.capitalize(),
                        'Средняя': f"{season_data['mean']:.1f}°C",
                        'Стд. откл.': f"{season_data['std']:.1f}°C",
                        'Минимум': f"{season_data['min']:.1f}°C",
                        'Максимум': f"{season_data['max']:.1f}°C"
                    })
            
            if seasons_data:
                display_df = pd.DataFrame(seasons_data)
                st.dataframe(display_df, use_container_width=True)
            
            st.markdown("**Климатическая норма:**")
            climate_norms = seasonal_temperatures[selected_city]
            for season, temp in climate_norms.items():
                st.write(f"{season.capitalize()}: {temp}°C")
    
    with tab2:
        st.header(f"📈 Анализ температурных трендов в {selected_city}")
        
        fig_yearly = create_yearly_trend_chart(yearly_stats, selected_city)
        st.plotly_chart(fig_yearly, use_container_width=True)
        
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
                    name=season.capitalize()
                ))
        
        fig_seasonal.update_layout(
            title=f'Средняя температура по сезонам',
            xaxis_title='Год',
            yaxis_title='Температура (°C)',
            height=400
        )
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            fig_hist = px.histogram(
                city_data, x='temperature', nbins=50,
                title='Распределение температур'
            )
            fig_hist.add_vline(x=overall_stats['mean'], line_dash="dash", line_color="red")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_stat2:
            heatmap_data = city_data.pivot_table(
                index='year',
                columns='month',
                values='temperature',
                aggfunc='mean'
            ).fillna(heatmap_data.mean() if 'heatmap_data' in locals() else 0)
            
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                          'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
            
            try:
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Месяц", y="Год", color="Температура (°C)"),
                    x=month_names,
                    color_continuous_scale='RdBu_r'
                )
                fig_heatmap.update_layout(height=400, title='Температурная карта')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except:
                st.info("Недостаточно данных для тепловой карты")
    
    with tab3:
        st.header("🌤️ Текущая погода")
        
        if st.button("Получить текущую погоду", type="primary"):
            if not st.session_state.api_key_valid:
                st.error("Сначала проверьте API ключ")
            else:
                with st.spinner("Запрос данных..."):
                    start_time = time.time()
                    
                    if method == "Синхронный":
                        weather_data = get_current_weather_sync(api_key_input, selected_city)
                    else:
                        async def get_async_weather():
                            results = await get_multiple_weather_async(api_key_input, [selected_city])
                            return results[0] if results else {'success': False, 'error': 'No results'}
                        
                        weather_data = asyncio.run(get_async_weather())
                    
                    request_time = time.time() - start_time
                    
                    if weather_data['success']:
                        st.session_state['weather_data'] = weather_data
                        st.session_state['request_time'] = request_time
                        st.success(f"Данные получены за {request_time:.2f} сек")
                    else:
                        st.error(f"Ошибка: {weather_data['error']}")
        
        if 'weather_data' in st.session_state and st.session_state['weather_data']['success']:
            weather = st.session_state['weather_data']
            
            cols_weather = st.columns(4)
            with cols_weather[0]:
                st.metric("🌡️ Температура", f"{weather['temperature']}°C")
            with cols_weather[1]:
                st.metric("💨 Ощущается как", f"{weather['feels_like']}°C")
            with cols_weather[2]:
                st.metric("💧 Влажность", f"{weather['humidity']}%")
            with cols_weather[3]:
                st.metric("🔽 Давление", f"{weather['pressure']} hPa")
                st.info(f"**🌤️ Погода:** {weather['description'].capitalize()}")
            
            current_temp = weather['temperature']
            hist_mean = overall_stats['mean']
            hist_std = overall_stats['std']
            deviation = current_temp - hist_mean
            z_score = deviation / hist_std if hist_std > 0 else 0
            
            if abs(z_score) <= 2:
                status, color = "✅ В пределах нормы", "green"
            elif abs(z_score) <= 3:
                status, color = "⚠️ Нестандартная", "orange"
            else:
                status, color = "🚨 Аномальная", "red"
            
            st.markdown(f"""
            <div style="background-color:{color}20; padding:15px; border-radius:10px; border-left:5px solid {color};">
                <h4>{status}</h4>
                <p><b>Текущая:</b> {current_temp}°C | <b>Средняя:</b> {hist_mean}°C</p>
                <p><b>Отклонение:</b> {deviation:+.1f}°C | <b>Z-оценка:</b> {z_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("⚡ Сравнение скорости асинхронных и синхронных запросов")
        
        if not st.session_state.api_key_valid:
            st.warning("Требуется действительный API ключ для тестирования")
        elif st.button("Запустить тест скорости", type="primary", use_container_width=True):
            test_cities = ["Berlin", "Paris", "London", "Tokyo", "Moscow", "New York", "Beijing"]
            
            st.info(f"Тестирование для {len(test_cities)} городов: {', '.join(test_cities)}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
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
                time.sleep(0.1)  # Небольшая пауза между запросами
            
            sync_total_time = time.time() - start_time
            
            start_time = time.time()
            
            async def run_async_test():
                return await get_multiple_weather_async(api_key_input, test_cities)
            
            async_results = asyncio.run(run_async_test())
            
            for i in range(len(test_cities)):
                progress_bar.progress((len(test_cities) + i + 1) / (len(test_cities) * 2))
            
            async_total_time = time.time() - start_time
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.metric(
                    "Синхронные запросы", 
                    f"{sync_total_time:.2f} сек",
                    f"{sync_total_time/len(test_cities):.2f} сек/город",
                    delta_color="normal"
                )
            
            with col_perf2:
                st.metric(
                    "Асинхронные запросы", 
                    f"{async_total_time:.2f} сек",
                    f"{async_total_time/len(test_cities):.2f} сек/город",
                    delta_color="normal"
                )
            
            st.markdown("---")
            st.subheader("Результаты сравнения")
            
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
                    text='Общее время (сек)',
                    color_discrete_map={'Синхронный': '#FF6B6B', 'Асинхронный': '#4ECDC4'}
                )
                
                fig_perf.update_traces(texttemplate='%{text:.2f} сек', textposition='outside')
                fig_perf.update_layout(
                    height=400, 
                    showlegend=False,
                    yaxis_title='Время (секунды)',
                    xaxis_title='Метод запроса'
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
                
                if speedup > 1.2:
                    st.success(f"**Асинхронные запросы быстрее синхронных в {speedup:.1f} раза!**")
                    st.markdown(f"""
                    **Ключевые показатели:**
                    - Синхронные запросы: {sync_total_time:.2f} секунд
                    - Асинхронные запросы: {async_total_time:.2f} секунд
                    - Экономия времени: {sync_total_time - async_total_time:.2f} секунд ({((sync_total_time - async_total_time)/sync_total_time*100):.0f}%)
                    - Среднее время на запрос: {sync_total_time/len(test_cities):.2f} сек (синхрон) vs {async_total_time/len(test_cities):.2f} сек (асинхрон)
                    
                    **Выводы:**
                    - Асинхронные запросы значительно эффективнее для одновременного получения данных по нескольким городам
                    """)
                else:
                    st.info(f"**Асинхронные запросы быстрее синхронных в {speedup:.1f} раза**")
                    st.markdown(f"""
                    **Ключевые показатели:**
                    - Синхронные запросы: {sync_total_time:.2f} секунд
                    - Асинхронные запросы: {async_total_time:.2f} секунд
                    - Разница во времени: {abs(sync_total_time - async_total_time):.2f} секунд
                    
                    **Выводы:**
                    - Для небольшого количества городов разница между методами незначительна
                    - Синхронный метод проще в реализации и отладке
                    """)

if __name__ == "__main__":
    main()
