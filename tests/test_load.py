from locust import HttpUser, task, between
import random
import string

def generate_random_url():
    domains = ["google.com", "github.com", "stackoverflow.com", "python.org"]
    path = ''.join(random.choices(string.ascii_lowercase, k=8))
    return f"https://{random.choice(domains)}/{path}"

def generate_random_alias():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

class URLShortenerUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:8000"
    
    def on_start(self):
        self.username = f"loadtest_{random.randint(1, 9999)}"
        self.password = "testpass123"
        
        self.client.post("/auth/register", json={
            "username": self.username,
            "email": f"{self.username}@test.com",
            "password": self.password})
        
        response = self.client.post("/auth/token", data={
            "username": self.username,
            "password": self.password})
        self.token = response.json().get("access_token")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
        self.created_codes = []
    
    @task(3)
    def create_short_link(self):
        with self.client.post(
            "/links/shorten",
            json={"original_url": generate_random_url()},
            headers=self.headers,
            catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                self.created_codes.append(data["short_code"])
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(1)
    def create_link_with_alias(self):
        self.client.post(
            "/links/shorten",
            json={
                "original_url": generate_random_url(),
                "custom_alias": generate_random_alias()}, headers=self.headers)
    
    @task(2)
    def get_link_stats(self):
        if self.created_codes:
            code = random.choice(self.created_codes)
            self.client.get(f"/links/{code}/stats")
    
    @task(5)
    def health_check(self):
        self.client.get("/health")
    
    @task(1)
    def root_endpoint(self):
        self.client.get("/")
