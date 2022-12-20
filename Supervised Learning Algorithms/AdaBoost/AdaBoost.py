from sklearn.ensemble import AdaBoostClassifier
adb_model = AdaBoostClassifier()
adb_fit = adb_model.fit(X_train, Y_train)
adb_pred = adb_fit.predict(X_test)