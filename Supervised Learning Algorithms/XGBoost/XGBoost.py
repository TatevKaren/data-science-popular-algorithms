from xgboost import XGBRegressor
xgboost_model = XGBRegressor()
xgboost_fit = xgboost_model.fit(X_train, Y_train)
xgboost_pred = xgboost_fit.predict(X_test)
