import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';
import API_BASE_URL from '../services/api';

export default function Login() {
  const navigate = useNavigate();

  const [userType, setUserType] = useState('noc');
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    setError('');

    if (!id.trim() || !password.trim()) {
      setError('Please fill in all fields');
      return;
    }

    try {
      setLoading(true);

      /*
       * Backend login endpoint
       *
       * Change /api/auth/login if your backend
       * uses a different login route.
       */
      const response = await fetch(
        `${API_BASE_URL}/api/auth/login`,
        {
          method: 'POST',

          headers: {
            'Content-Type': 'application/json',
          },

body: JSON.stringify({
        user_id: id,
        password,
        user_type: userType,
      }),
        }
      );

      const result = await response.json();

      console.log('Login API Response:', result);

      if (!response.ok) {
        throw new Error(
          result?.message || 'Invalid ID or password'
        );
      }

      /*
       * Expected successful response:
       *
       * {
       *   success: true,
       *   message: "Login successful",
       *   data: {
       *      token: "...",
       *      id: "...",
       *      userType: "noc"
       *   }
       * }
       */

      if (!result.success) {
        throw new Error(
          result.message || 'Login failed'
        );
      }

      /*
       * Support either:
       *
       * result.data.token
       *
       * or
       *
       * result.token
       */

      const token =
        result?.data?.token ||
        result?.token;

      if (!token) {
        throw new Error(
          'Login successful, but authentication token was not returned by the server.'
        );
      }

      /*
       * Save authentication information
       */

      localStorage.setItem(
        'authToken',
        token
      );

      localStorage.setItem(
        'user',
        JSON.stringify(
          result.data || result
        )
      );

      localStorage.setItem(
        'userType',
        userType
      );

      /*
       * Redirect based on role
       */

      if (userType === 'admin') {
        navigate('/admin');
      } else {
        navigate('/dashboard');
      }

    } catch (err) {
      console.error(
        'CTS Login Error:',
        err
      );

      /*
       * Handle network error separately
       */

      if (
        err instanceof TypeError &&
        err.message.includes('fetch')
      ) {
        setError(
          'Unable to connect to CTS backend. Make sure the backend is running on port 8000.'
        );
      } else {
        setError(
          err.message ||
          'Login failed. Please try again.'
        );
      }

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-wrapper">

      {/* =====================================================
          LOGIN CONTAINER
      ====================================================== */}

      <div className="login-container">

        {/* ===================================================
            LEFT SECTION
        ==================================================== */}

        <div className="login-left">

          <div className="login-branding">

            <div className="logo-circle">
              <span className="logo-text">
                CTS
              </span>
            </div>

            <h2 className="brand-title">
              Incident Management System
            </h2>

            <p className="brand-subtitle">
              Efficient ticket resolution and technical support
            </p>

          </div>


          <div className="login-features">

            <div className="feature-item">

              <div className="feature-icon">
                ⚡
              </div>

              <p>
                Fast Incident Resolution
              </p>

            </div>


            <div className="feature-item">

              <div className="feature-icon">
                🔍
              </div>

              <p>
                AI-Powered Analysis
              </p>

            </div>


            <div className="feature-item">

              <div className="feature-icon">
                👥
              </div>

              <p>
                Team Collaboration
              </p>

            </div>

          </div>

        </div>


        {/* ===================================================
            RIGHT SECTION
        ==================================================== */}

        <div className="login-right">

          <div className="login-form-wrapper">

            {/* Header */}

            <div className="form-header">

              <h1 className="form-title">
                Welcome Back
              </h1>

              <p className="form-subtitle">
                Sign in to your account
              </p>

            </div>


            {/* =================================================
                ROLE SELECTION
            ================================================== */}

            <div className="role-selector">

              <button
                type="button"
                className={`role-btn ${
                  userType === 'noc'
                    ? 'active'
                    : ''
                }`}
                onClick={() => {
                  setUserType('noc');
                  setError('');
                }}
                disabled={loading}
              >
                <span className="role-icon">
                  📋
                </span>

                <span>
                  NOC
                </span>
              </button>


              <button
                type="button"
                className={`role-btn ${
                  userType === 'admin'
                    ? 'active'
                    : ''
                }`}
                onClick={() => {
                  setUserType('admin');
                  setError('');
                }}
                disabled={loading}
              >
                <span className="role-icon">
                  ⚙️
                </span>

                <span>
                  Admin
                </span>
              </button>

            </div>


            {/* =================================================
                ERROR
            ================================================== */}

            {error && (

              <div className="error-alert">
                {error}
              </div>

            )}


            {/* =================================================
                LOGIN FORM
            ================================================== */}

            <form
              onSubmit={handleSubmit}
              className="form"
            >

              {/* Employee ID */}

              <div className="form-group">

                <label
                  htmlFor="id"
                  className="form-label"
                >
                  Employee ID
                </label>

                <div className="input-wrapper">

                  <span className="input-icon">
                    👤
                  </span>

                  <input
                    type="text"
                    id="id"
                    name="id"
                    value={id}
                    onChange={(e) =>
                      setId(e.target.value)
                    }
                    placeholder="Enter your ID"
                    className="form-input"
                    autoComplete="username"
                    disabled={loading}
                    required
                  />

                </div>

              </div>


              {/* Password */}

              <div className="form-group">

                <label
                  htmlFor="password"
                  className="form-label"
                >
                  Password
                </label>

                <div className="input-wrapper">

                  <span className="input-icon">
                    🔒
                  </span>

                  <input
                    type={
                      showPassword
                        ? 'text'
                        : 'password'
                    }
                    id="password"
                    name="password"
                    value={password}
                    onChange={(e) =>
                      setPassword(e.target.value)
                    }
                    placeholder="Enter your password"
                    className="form-input"
                    autoComplete="current-password"
                    disabled={loading}
                    required
                  />

                  <button
                    type="button"
                    className="toggle-password"
                    onClick={() =>
                      setShowPassword(
                        !showPassword
                      )
                    }
                    disabled={loading}
                    aria-label={
                      showPassword
                        ? 'Hide password'
                        : 'Show password'
                    }
                  >
                    {showPassword
                      ? '👁️'
                      : '👁️‍🗨️'}
                  </button>

                </div>

              </div>


              {/* Submit */}

              <button
                type="submit"
                className="submit-btn"
                disabled={loading}
              >

                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Signing in...
                  </>
                ) : (
                  'Sign In'
                )}

              </button>

            </form>


            {/* Footer */}

            <div className="form-footer">

              <button
                type="button"
                className="forgot-link"
                onClick={() =>
                  setError(
                    'Please contact the CTS administrator to reset your password.'
                  )
                }
              >
                Forgot password?
              </button>

            </div>

          </div>

        </div>

      </div>


      {/* =====================================================
          FOOTER
      ====================================================== */}

      <div className="login-footer">

        <p>
          © 2026 Customer Technical Support.
          All rights reserved.
        </p>

      </div>

    </div>
  );
}