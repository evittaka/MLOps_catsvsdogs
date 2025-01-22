from locust import HttpUser, between, task


class LocustTest(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task(3)
    def post_image_for_prediction(self):
        """Task to post an image to the prediction endpoint."""
        files = {"data": ("tests/performance_test/cat.jpg", open("tests/performance_test/cat.jpg", "rb"), "image/jpg")}

        self.client.post(url="/predict", headers={"accept": "application/json"}, files=files)
