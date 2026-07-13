from predictor import predict_url

urls = [

    "https://google.com",

    "https://facebook.com",

    "https://github.com",

    "http://paypal-login-security.xyz",

    "http://verify-account-paypal.com",

    "http://secure-bank-login.xyz"

]

for url in urls:

    result = predict_url(url)

    print("=" * 50)

    print(url)

    print(result)