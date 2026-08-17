# CTS Backend - Login System Setup

## Installation

1. Install required dependencies:
```bash
pip install flask flask-cors pyjwt python-dotenv
```

2. Create a `.env` file in the `RAG` directory:
```
JWT_SECRET=your-secret-key-here
FRONTEND_URL=http://localhost:5173
PORT=5000
DEBUG=True
DATABASE_URL=sqlite:///cts.db
```

## Project Structure

```
backend/
├── controllers/
│   ├── __init__.py
│   └── login_controllers.py      # Request handlers
├── services/
│   ├── __init__.py
│   └── login_services.py         # Business logic
├── routes/
│   ├── __init__.py
│   └── login_routes.py           # API endpoints
├── utils/
├── config.py                      # Configuration
└── main.py                        # Flask app setup
```

## Running the Backend

```bash
python RAG/backend/main.py
```

The backend will start on `http://localhost:5000`

## API Endpoints

### 1. Login
**POST** `/api/auth/login`

Request:
```json
{
    "user_id": "NOC001",
    "password": "password123",
    "user_type": "noc"
}
```

Response (Success):
```json
{
    "success": true,
    "message": "Login successful",
    "data": {
        "user_id": "NOC001",
        "user_type": "noc",
        "name": "John NOC",
        "email": "john@noc.com",
        "token": "eyJhbGciOiJIUzI1NiIs...",
        "expires_in": 86400
    }
}
```

### 2. Verify Token
**GET** `/api/auth/verify-token`

Headers:
```
Authorization: Bearer <token>
```

Response:
```json
{
    "success": true,
    "message": "Token is valid",
    "data": {
        "user_id": "NOC001",
        "user_type": "noc",
        "name": "John NOC",
        "email": "john@noc.com"
    }
}
```

### 3. Logout
**POST** `/api/auth/logout`

Headers:
```
Authorization: Bearer <token>
```

Response:
```json
{
    "success": true,
    "message": "Logout successful"
}
```

### 4. Protected Example Route
**GET** `/api/auth/protected-example`

Headers:
```
Authorization: Bearer <token>
```

## Sample Login Credentials (for testing)

### NOC Users
- ID: `NOC001` | Password: `password123`
- ID: `NOC002` | Password: `password123`

### Admin Users
- ID: `ADMIN001` | Password: `admin123`

## Key Features

✅ User authentication with JWT tokens
✅ Role-based access (NOC/Admin)
✅ Token verification and expiration
✅ Protected route decorator
✅ Error handling
✅ CORS enabled
✅ Environment-based configuration

## TODO - Database Integration

Currently using hardcoded sample data. To integrate with a real database:

1. Replace `_get_user_from_db()` in `login_services.py` with actual database queries
2. Implement proper password hashing (consider using `werkzeug.security`)
3. Add database models for User and UserCredentials
4. Update requirements.txt with database driver (e.g., `sqlalchemy`)

## Error Handling

The API returns consistent error responses:

```json
{
    "success": false,
    "message": "Error description here"
}
```

HTTP Status Codes:
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Server Error
