import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,r2_score, mean_absolute_error,mean_squared_error 
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import base64
from sklearn.decomposition import PCA
import numpy as np
import category_encoders as ce
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder


# Tiêu đề của ứng dụng
st.title("Phân tích dữ liệu và dự đoán với Linear Regression, Logistic Regression, KNN và Decision Tree")

# Tiếp nhận file CSV từ người dùng
uploaded_file = st.file_uploader("Chọn một file CSV", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(uploaded_file)
    
    # Hiển thị 5 dòng đầu tiên của dữ liệu
    st.subheader("Dữ liệu đầu tiên")
    st.write(df.head())
    
    # Thống kê cơ bản
    st.subheader("Thống kê cơ bản")
    st.write(f"Số lượng dòng: {df.shape[0]}")
    st.write(f"Số lượng cột: {df.shape[1]}")
    st.write(df.describe())
    st.sidebar.subheader("Gợi ý mô hình")
    chosen_column = st.sidebar.selectbox("Chọn một cột để gợi ý mô hình", df.columns)

    if st.sidebar.button("Gợi ý"):
        try:
            # Xác định loại dữ liệu của cột
            data_type = df[chosen_column].dtype

            # Gợi ý mô hình dựa vào loại dữ liệu và thông tin phân phối
            if set(df[chosen_column].unique()) == {0, 1}:
                st.sidebar.write(f"Cột '{chosen_column}' đã được mã hóa thành 0 và 1.")
                st.sidebar.write("Bạn có thể sử dụng mô hình Logistic Regression, KNN, Random Forest.")
            elif data_type == 'int64' or data_type == 'float64':
                st.sidebar.write(f"Cột '{chosen_column}' là dạng số.")
                st.sidebar.write("Bạn có thể sử dụng mô hình Linear Regression hoặc Decision Tree.")
                # Thêm thông tin phân phối dữ liệu
                mean_value = df[chosen_column].mean()
                st.sidebar.write(f"Giá trị trung bình của '{chosen_column}': {mean_value:.2f}")
                
                std_value = df[chosen_column].std()
                st.sidebar.write(f"Độ lệch chuẩn của '{chosen_column}': {std_value:.2f}")

            elif data_type == 'object':
                st.sidebar.write(f"Cột '{chosen_column}' là dạng categorical.")
                st.sidebar.write("Bạn có thể sử dụng mô hình Logistic Regression, KNN, Random Forest")
                
                # Thêm thông tin về số lượng giá trị duy nhất
                unique_count = df[chosen_column].nunique()
                st.sidebar.write(f"Số lượng giá trị duy nhất của '{chosen_column}': {unique_count}")
       
        except KeyError as e:
            st.sidebar.write(f"Cột '{chosen_column}' không tồn tại trong dữ liệu.")

    # Chọn thuật toán và tính năng
    st.sidebar.title("Chọn tính năng")
    feature = st.sidebar.radio("Tính năng", ["Clean Data", "Dự đoán", "Data Analysis"])
    
    if feature == "Dự đoán":
        # Chọn các cột để hiển thị heatmap
        st.subheader("Chọn các cột để hiển thị Confusion matrix")
        chosen_columns = st.multiselect("Chọn các cột", df.columns)
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[chosen_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Ma trận tương quan")
        st.pyplot(plt)
        ml_algorithm = st.sidebar.selectbox("Chọn thuật toán", ["Linear Regression", "Logistic Regression", "KNN", "Decision Tree","Random Forest"])
        
        if ml_algorithm == "Linear Regression":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
             #Hiển thị form nhập giá trị để dự đoán
            st.sidebar.subheader("Nhập giá trị muốn dự đoán")
            input_values = {}
            for var in independent_vars:
                input_values[var] = st.sidebar.number_input(f"{var}:", step=1.0)

            if st.sidebar.button("Dự đoán"):
                try:
                        # Tạo DataFrame từ giá trị đầu vào của người dùng
                    input_data = pd.DataFrame([input_values])
                    X = df[independent_vars]
                    y = df[dependent_var]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    scores = cross_val_score(model, X, y, cv=2, scoring='r2')

                    # Display cross-validation R^2 scores
                    st.write(f"Cross-Validation R^2 Scores: {scores}")
                    st.write(f"Mean CV Score: {scores.mean()}")
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    predicted_value = model.predict(input_data)

                    # Hiển thị kết quả dự đoán
                    st.subheader("Kết quả giá trị muốn dự đoán")
                    st.write(predicted_value)
                    # Hiển thị kết quả dự đoán vs thực tế
                    st.subheader("Kết quả dự đoán vs Giá trị thực tế")
                    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                    st.write(result_df)

                    # Trực quan hóa dự đoán vs thực tế
                    st.subheader("Biểu đồ dự đoán vs Thực tế")
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=y_test, y=y_pred)
                    sns.lineplot(x=y_test, y=y_test, color='red', label='Đường thẳng tuyến tính')
                    plt.xlabel("Thực tế")
                    plt.ylabel("Dự đoán")
                    plt.title("Biểu đồ dự đoán vs Thực tế")
                    plt.legend()
                    st.pyplot(plt)

                    # Biểu đồ phân phối của dự đoán và thực tế
                    st.subheader("Biểu đồ phân phối của Dự đoán và Thực tế")
                    plt.figure(figsize=(10, 6))
                    sns.kdeplot(y_test, label='Thực tế', color='blue', shade=True)
                    sns.kdeplot(y_pred, label='Dự đoán', color='red', shade=True)
                    plt.xlabel("Giá trị")
                    plt.ylabel("Mật độ")
                    plt.title("Phân phối của Dự đoán và Thực tế")
                    plt.legend()
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")

        elif ml_algorithm == "Logistic Regression":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
            st.sidebar.subheader("Nhập giá trị muốn dự đoán")
            input_values = {}
            for var in independent_vars:
                input_values[var] = st.sidebar.number_input(f"{var}:", step=1.0)

            if st.sidebar.button("Dự đoán"):
                try:
                    X = df[independent_vars]
                    y = df[dependent_var]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                                # Tạo DataFrame từ giá trị đầu vào của người dùng
                    input_data = pd.DataFrame([input_values])
                    y_pred = model.predict(X_test)
                    predicted_value = model.predict(input_data)

                    # Hiển thị kết quả dự đoán
                    st.subheader("Kết quả giá trị muốn dự đoán")
                    st.write(predicted_value)
                    # Hiển thị độ chính xác và các chỉ số đánh giá khác
                    st.subheader("Độ chính xác và Chỉ số đánh giá")
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    st.write(f"Độ chính xác: {accuracy:.2f}")
                    st.write(f"Precision: {precision:.2f}")
                    st.write(f"Recall: {recall:.2f}")
                    st.write(f"F1 Score: {f1:.2f}")

                    # Hiển thị kết quả dự đoán
                    st.subheader("Kết quả dự đoán")
                    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                    st.write(result_df)

                    # Ma trận Confusion
                    st.subheader("Ma trận Confusion")
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
                    plt.title("Ma trận Confusion")
                    plt.xlabel("Dự đoán")
                    plt.ylabel("Thực tế")
                    st.pyplot(plt)

        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")
        elif ml_algorithm == "KNN":
            
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
            st.sidebar.subheader("Nhập giá trị muốn dự đoán")
            input_values = {}
            for var in independent_vars:
                input_values[var] = st.sidebar.number_input(f"{var}:", step=1.0)

            if st.sidebar.button("Dự đoán"):
                X = df[independent_vars]
                y = df[dependent_var]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
                input_data = pd.DataFrame([input_values])

                y_pred = model.predict(X_test)
                predicted_value = model.predict(input_data)

                    # Hiển thị kết quả dự đoán
                st.subheader("Kết quả giá trị muốn dự đoán")
                st.write(predicted_value)
                # Hiển thị độ chính xác và các chỉ số đánh giá khác
                st.subheader("Độ chính xác và Chỉ số đánh giá")
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                st.write(f"Độ chính xác: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Hiển thị kết quả dự đoán
                st.subheader("Kết quả dự đoán")
                result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                st.write(result_df)

                # Ma trận Confusion
                st.subheader("Ma trận Confusion")
                plt.figure(figsize=(10, 6))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
                plt.title("Ma trận Confusion")
                plt.xlabel("Dự đoán")
                plt.ylabel("Thực tế")
                st.pyplot(plt)

        elif ml_algorithm == "Decision Tree":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
            max_depth = st.sidebar.slider("Chọn độ sâu tối đa của cây", 1, 20, 5)
            min_samples_split = st.sidebar.slider("Số lượng mẫu tối thiểu để tách nút", 2, 20, 2)
            min_samples_leaf = st.sidebar.slider("Số lượng mẫu tối thiểu tại một lá", 1, 20, 1)
            st.sidebar.subheader("Nhập giá trị muốn dự đoán")
            input_values = {}
            for var in independent_vars:
                input_values[var] = st.sidebar.number_input(f"{var}:", step=1.0)

            if st.sidebar.button("Dự đoán"):
                try:
                    X = df[independent_vars]
                    y = df[dependent_var]
                    input_data = pd.DataFrame([input_values])

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    model.fit(X_train, y_train)
                    feature_importances = model.feature_importances_
                    st.subheader("Feature Importances")
                    feature_importance_df = pd.DataFrame({'Feature': independent_vars, 'Importance': feature_importances})
                    st.write(feature_importance_df.sort_values(by='Importance', ascending=False))
                    y_pred = model.predict(X_test)
                    predicted_value = model.predict(input_data)

                    # Hiển thị kết quả dự đoán
                    st.subheader("Kết quả giá trị muốn dự đoán")
                    st.write(predicted_value)
                    # Hiển thị độ chính xác và các chỉ số đánh giá khác
                    st.subheader("Độ chính xác và Chỉ số đánh giá")
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    st.write(f"Độ chính xác: {accuracy:.2f}")
                    st.write(f"Precision: {precision:.2f}")
                    st.write(f"Recall: {recall:.2f}")
                    st.write(f"F1 Score: {f1:.2f}")

                    # Hiển thị kết quả dự đoán
                    st.subheader("Kết quả dự đoán")
                    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                    st.write(result_df)

                    # Trực quan hóa Ma trận Confusion
                    st.subheader("Ma trận Confusion")
                    fig, ax = plt.subplots()
                    cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
                    cm_display.ax_.set_xlabel('Predicted labels')
                    cm_display.ax_.set_ylabel('True labels')
                    st.pyplot(fig)

                    # Trực quan hóa biểu đồ Feature Importances
                    st.subheader("Biểu đồ Feature Importances")
                    fig, ax = plt.subplots()
                    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False), ax=ax)
                    ax.set_xlabel('Importance')
                    ax.set_ylabel('Feature')
                    st.pyplot(fig)

                    # Trực quan hóa phân phối kết quả dự đoán
                    st.subheader("Phân phối kết quả dự đoán")
                    fig, ax = plt.subplots()
                    sns.histplot(y_test, label='Thực tế', color='blue', kde=True, ax=ax)
                    sns.histplot(y_pred, label='Dự đoán', color='red', kde=True, ax=ax)
                    plt.legend()
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)

                    # In ra cây quyết định
                    st.subheader("Cây quyết định")
                    plt.figure(figsize=(20, 10))
                    plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")

        elif ml_algorithm == "Random Forest":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
            n_estimators = st.sidebar.slider("Số lượng cây trong rừng", 10, 100, 50)
            max_depth = st.sidebar.slider("Chọn độ sâu tối đa của cây", 1, 20, 5)
            min_samples_split = st.sidebar.slider("Số lượng mẫu tối thiểu để tách nút", 2, 20, 2)
            min_samples_leaf = st.sidebar.slider("Số lượng mẫu tối thiểu tại một lá", 1, 20, 1)
            st.sidebar.subheader("Nhập giá trị muốn dự đoán")
            input_values = {}
            for var in independent_vars:
                input_values[var] = st.sidebar.number_input(f"{var}:", step=1.0)

            if st.sidebar.button("Dự đoán"):
                X = df[independent_vars]
                y = df[dependent_var]
                input_data = pd.DataFrame([input_values])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
                model.fit(X_train, y_train)
                feature_importances = model.feature_importances_
                st.subheader("Feature Importances")
                feature_importance_df = pd.DataFrame({'Feature': independent_vars, 'Importance': feature_importances})
                st.write(feature_importance_df.sort_values(by='Importance', ascending=False))
                y_pred = model.predict(X_test)
                predicted_value = model.predict(input_data)

                    # Hiển thị kết quả dự đoán
                st.subheader("Kết quả giá trị muốn dự đoán")
                st.write(predicted_value)
                # Hiển thị độ chính xác và các chỉ số đánh giá khác
                st.subheader("Độ chính xác và Chỉ số đánh giá")
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                st.write(f"Độ chính xác: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Hiển thị kết quả dự đoán
                st.subheader("Kết quả dự đoán")
                result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                st.write(result_df)

                # Trực quan hóa Ma trận Confusion
                st.subheader("Ma trận Confusion")
                fig, ax = plt.subplots()
                cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
                st.pyplot(fig)

                # Trực quan hóa biểu đồ Feature Importances
                st.subheader("Biểu đồ Feature Importances")
                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False), ax=ax)
                st.pyplot(fig)

                # Trực quan hóa phân phối kết quả dự đoán
                st.subheader("Phân phối kết quả dự đoán")
                fig, ax = plt.subplots()
                sns.histplot(y_test, label='Thực tế', color='blue', kde=True, ax=ax)
                sns.histplot(y_pred, label='Dự đoán', color='red', kde=True, ax=ax)
                plt.legend()
                st.pyplot(fig)
                                # Trực quan hóa các cây quyết định
                st.subheader("Hiển thị các cây quyết định trong rừng")
                for i in range(min(n_estimators, 5)):  # Chỉ hiển thị tối đa 5 cây để không làm quá tải giao diện
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(model.estimators_[i], feature_names=independent_vars, class_names=model.classes_.astype(str), filled=True, ax=ax)
                    st.pyplot(fig)
            
    elif feature == "Data Analysis":
        selected_columns = st.sidebar.multiselect("Chọn cột để phân tích", df.columns)
        chart_type = st.sidebar.selectbox("Chọn kiểu biểu đồ", ["Histogram", "Boxplot", "Scatterplot", "Bar", "Pie", "Confusion matrix", "Countplot","Area","Pairplot"])

        if st.sidebar.button("Hiển thị biểu đồ"):
            if chart_type == "Histogram":
                for col in selected_columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col], kde=True)
                    plt.title(f"Biểu đồ Histogram của {col}")
                    plt.xlabel(col)
                    plt.ylabel("Tần suất")
                    st.pyplot(plt)
            elif chart_type == "Boxplot":
                if len(selected_columns) < 1 or len(selected_columns) > 2:
                    st.sidebar.warning("Chọn 1 hoặc 2 cột để hiển thị biểu đồ boxplot.")
                else:
                    plt.figure(figsize=(10, 6))
                    if len(selected_columns) == 1:
                        sns.boxplot(x=df[selected_columns[0]])
                        plt.title(f"Biểu đồ Boxplot của {selected_columns[0]}")
                        plt.xlabel(selected_columns[0])
                    elif len(selected_columns) == 2:
                        sns.boxplot(data=df, x=selected_columns[0], y=selected_columns[1])
                        plt.title(f"Biểu đồ Boxplot giữa {selected_columns[0]} và {selected_columns[1]}")
                        plt.xlabel(selected_columns[0])
                        plt.ylabel(selected_columns[1])
                    st.pyplot(plt)

            elif chart_type == "Scatterplot":
                if len(selected_columns) == 2:
                    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1])
                    st.plotly_chart(fig)
                    st.sidebar.warning("Chọn 1 hoặc 2 cột để hiển thị scatterplot.")
                else:
                    plt.figure(figsize=(10, 6))
                    if len(selected_columns) == 1:
                        # Sử dụng index làm trục x và giá trị cột làm trục y
                        sns.scatterplot(x=df.index, y=df[selected_columns[0]])
                        plt.title(f"Biểu đồ Scatterplot của {selected_columns[0]}")
                        plt.xlabel("Index")
                        plt.ylabel(selected_columns[0])
                    else:
                        x_column, y_column = selected_columns
                        sns.scatterplot(x=df[x_column], y=df[y_column])
                        plt.title(f"Biểu đồ Scatterplot giữa {x_column} và {y_column}")
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    st.pyplot(plt)
            elif chart_type == "Pie":
                if len(selected_columns) != 1:
                    st.sidebar.warning("Chọn đúng 1 cột để hiển thị biểu đồ pie.")
                else:
                    pie_column = selected_columns[0]
                    pie_data = df[pie_column].value_counts()
                    plt.figure(figsize=(10, 6))
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
                    plt.title(f"Biểu đồ Pie của {pie_column}")
                    st.pyplot(plt)
            elif chart_type == "Confusion matrix":
                plt.figure(figsize=(12, 8))
                correlation_matrix = df[selected_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title("Ma trận tương quan")
                st.pyplot(plt)
            elif chart_type == "Bar":
                if len(selected_columns) < 1 or len(selected_columns) > 2:
                    st.sidebar.warning("Chọn 1 hoặc 2 cột để hiển thị biểu đồ bar.")
                else:
                    plt.figure(figsize=(10, 6))
                    if len(selected_columns) == 1:
                        # Sử dụng value_counts() để lấy số lần xuất hiện của mỗi giá trị
                        data = df[selected_columns[0]].value_counts()
                        sns.barplot(x=data.index, y=data.values)
                        plt.title(f"Biểu đồ Bar của {selected_columns[0]}")
                        plt.xlabel(selected_columns[0])
                        plt.ylabel("Số lượng")
                    else:
                        # Sử dụng hai cột: một cho trục x và một cho trục y
                        sns.barplot(x=df[selected_columns[0]], y=df[selected_columns[1]])
                        plt.title(f"Biểu đồ Bar giữa {selected_columns[0]} và {selected_columns[1]}")
                        plt.xlabel(selected_columns[0])
                        plt.ylabel(selected_columns[1])
                    st.pyplot(plt)

            elif chart_type == "Countplot":
                if len(selected_columns) != 1:
                    st.sidebar.warning("Chọn đúng 1 cột để hiển thị biểu đồ countplot.")
                else:
                    countplot_column = selected_columns[0]
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x=df[countplot_column])
                    plt.title(f"Biểu đồ Countplot của {countplot_column}")
                    plt.xlabel(countplot_column)
                    plt.ylabel("Số lượng")
                    st.pyplot(plt)
                    
            elif chart_type == "Pairplot":
                if len(selected_columns) < 2:
                    st.warning("Select at least two columns for pairplot.")
                else:
                    plt.figure(figsize=(12, 8))
                    sns.pairplot(df[selected_columns])
                    plt.title("Pairplot")
                    st.pyplot(plt)

            elif chart_type == "Area":
                if len(selected_columns) != 1:
                    st.warning("Select one column for the area plot.")
                else:
                    plt.figure(figsize=(12, 8))
                    sns.lineplot(data=df, x=df.index, y=selected_columns[0], label=selected_columns[0])
                    plt.fill_between(df.index, df[selected_columns[0]], color="skyblue", alpha=0.3)
                    plt.title(f"Area Plot of {selected_columns[0]}")
                    plt.xlabel("Time")
                    plt.ylabel("Value")
                    plt.legend()
                    st.pyplot(plt)

         
            # Chức năng Clean Data
    # Chức năng Clean Data
    elif feature == "Clean Data":
        st.subheader("Thông tin dữ liệu trước khi làm sạch")
        st.write(df.head(10))

        st.subheader("Thống kê dữ liệu trước khi làm sạch")
        st.write(df.describe())
        # Hiển thị kích thước của DataFrame
        st.write("Kích thước của dữ liệu (số dòng, số cột):", df.shape)

        # Hiển thị thông tin về các cột và kiểu dữ liệu
        st.write("Thông tin về các cột và kiểu dữ liệu:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Hiển thị tên các cột
        st.write("Tên các cột trong dữ liệu:", df.columns.tolist())

        # Đếm số lượng giá trị null trong mỗi cột
        null_counts = df.isnull().sum()

        # Hiển thị số lượng giá trị null
        st.write("Số lượng giá trị null trong mỗi cột:")
        st.write(null_counts)
        # Phần xử lý dữ liệu null
        # Hiển thị chỉ các cột có giá trị null để sửa
        st.subheader("Các cột chứa giá trị null")
        # Hiển thị chỉ các cột có giá trị null để sửa
        st.subheader("Các cột chứa giá trị null")

                # Hiển thị chỉ các cột có giá trị null để sửa
        st.subheader("Các cột chứa giá trị null")

        null_columns = df.columns[df.isnull().any()]  # Lọc các cột có giá trị null

        for column in null_columns:
            st.write(f"### Cột {column}:")
            null_handling_method = st.selectbox(
                f"Chọn phương pháp xử lý null cho cột {column}:",
                ["NULL","Xóa cột", "Điền giá trị trung bình của cột",
                "Điền giá trị trung vị của cột", "Điền giá trị mode của cột"]
            )

            if null_handling_method == "Xóa cột":
                # Xóa các dòng có giá trị null trong cột
                df[column] = df[column].dropna()
                st.write(f"Đã xóa cột có giá trị null trong cột {column}.")

            elif null_handling_method == "Điền giá trị trung bình của cột":
                # Điền giá trị trung bình của cột cho các giá trị null
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
                st.write(f"Đã điền giá trị trung bình của cột {column} cho các giá trị null.")

            elif null_handling_method == "Điền giá trị trung vị của cột":
                # Điền giá trị trung vị của cột cho các giá trị null
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                st.write(f"Đã điền giá trị trung vị của cột {column} cho các giá trị null.")

            elif null_handling_method == "Điền giá trị mode của cột":
                # Điền giá trị mode của cột cho các giá trị null
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)
                st.write(f"Đã điền giá trị mode của cột {column} cho các giá trị null.")

               # Hiển thị dữ liệu sau khi xử lý null
        st.subheader("Dữ liệu sau khi xử lý null")
        st.write(df)
        null_counts = df.isnull().sum()

        # Hiển thị số lượng giá trị null
        st.write("Số lượng giá trị null trong mỗi cột:")
        st.write(null_counts)
        # Chức năng xóa cột
        st.subheader("Xóa cột")
        columns_to_drop = st.multiselect("Chọn cột để xóa:", df.columns)
        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
            st.success("Đã xóa các cột được chọn.")

        # Chức năng xóa dòng có giá trị thiếu
        st.subheader("Xóa dòng có giá trị thiếu")
        if st.button("Xóa dòng có giá trị thiếu"):
            df = df.dropna()
            st.success("Đã xóa các dòng có giá trị thiếu.")
        # Chức năng xử lý dữ liệu trùng lặp
        st.subheader("Xử lý dữ liệu trùng lặp")
        if st.button("Loại bỏ các hàng trùng lặp"):
            initial_rows = df.shape[0]  # Số hàng ban đầu
            df.drop_duplicates(inplace=True)  # Loại bỏ các hàng trùng lặp
            final_rows = df.shape[0]  # Số hàng sau khi loại bỏ
            removed_rows = initial_rows - final_rows
            st.success(f"Đã loại bỏ {removed_rows} hàng trùng lặp.")
            st.write("Số lượng hàng sau khi loại bỏ:", final_rows)
                # Hiển thị thông tin về các cột số
        st.subheader("Thông tin về các cột số:")
        st.write(df.select_dtypes(include=['float', 'int']).describe())
            # Chức năng xử lý dữ liệu bị nhiễu
        st.subheader("Xử lý dữ liệu bị nhiễu")
        for column in df.select_dtypes(include=['float', 'int']).columns:
            st.write(f"### Cột {column}:")
            outlier_method = st.selectbox(
                f"Chọn phương pháp xử lý outliers cho cột {column}:",
                ["Giữ nguyên", "Xóa outliers"]
            )
            if outlier_method == "Giữ nguyên":
                st.write(f"Đã giữ nguyên các giá trị outliers trong cột {column}.")
                
            elif outlier_method == "Xóa outliers":
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                st.write(f"Đã xóa các giá trị outliers trong cột {column}.")
            
        # Hiển thị dữ liệu sau khi xử lý outliers
        st.subheader("Dữ liệu sau khi xử lý outliers")
        st.write(df)
        # Chức năng sửa đổi kiểu dữ liệu
        st.subheader("Sửa đổi kiểu dữ liệu")
        date_format = "%Y-%m-%d"  # Định dạng ngày tháng, có thể thay đổi tùy theo nhu cầu
        for column in df.columns:
            new_data_type = st.selectbox(
                f"Chọn kiểu dữ liệu mới cho cột {column}:",
                ["int", "float", "object", "boolean", "datetime"]
            )
            if new_data_type != df[column].dtype.name:
                try:
                    if new_data_type == "int":
                        df[column] = df[column].astype(int)
                    elif new_data_type == "float":
                        df[column] = df[column].astype(float)
                    elif new_data_type == "object":
                        df[column] = df[column].astype(str)
                    elif new_data_type == "boolean":
                        df[column] = df[column].astype(bool)
                    elif new_data_type == "datetime":
                        df[column] = pd.to_datetime(df[column], format=date_format)
                    st.success(f"Đã chuyển cột {column} sang kiểu dữ liệu {new_data_type}.")
                except ValueError:
                    st.error(f"Không thể chuyển đổi cột {column} sang kiểu dữ liệu {new_data_type}.")

        # Hiển thị kiểu dữ liệu của các cột
        st.subheader("Kiểu dữ liệu của các cột")
        st.write(df.dtypes)
        # Thêm phần chuẩn hóa dữ liệu (Encoding)
        st.subheader("Chuẩn hóa dữ liệu (Encoding)")
        cat_columns = df.select_dtypes(include=['object']).columns
        for column in cat_columns:
            st.write(f"### Cột {column}:")
            encoding_method = st.selectbox(
                f"Chọn phương pháp encoding cho cột {column}:",
                ["Label Encoding"]
            )
            
            if encoding_method == "Label Encoding":
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column])
                st.write(f"Đã áp dụng Label Encoding cho cột {column}.")
           
        # Hiển thị dữ liệu sau khi chuẩn hóa
        st.subheader("Dữ liệu sau khi chuẩn hóa")
        st.write(df)

    
        # Cho phép người dùng tải về dataset đã được làm sạch
        st.subheader("Tải về dataset đã làm sạch")
        cleaned_data_file = df.to_csv(index=False)
        st.download_button("Tải về CSV", cleaned_data_file, "cleaned_data.csv", mime="text/csv", help="Nhấn vào đây để tải về dataset đã được làm sạch.")
        # Tiêu đề của ứng dụng
        st.title("Merge DataFrame")
                            # Định nghĩa hàm để đọc file CSV hoặc Excel# Định nghĩa hàm để đọc file CSV hoặc Excel
        def read_data(file):
            df = None
            if file.name.endswith('csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('xls', 'xlsx')):
                df = pd.read_excel(file)
            return df

        # Tải lên hai file CSV từ người dùng
        left_df = st.file_uploader("Chọn file cho DataFrame trái (left)", type=["csv", "xlsx"])
        right_df = st.file_uploader("Chọn file cho DataFrame phải (right)", type=["csv", "xlsx"])

        # Lưu ý: Bạn nên kiểm tra left_df và right_df xem chúng có None hay không trước khi đọc và hiển thị
        if left_df is not None and right_df is not None:
            # Đọc hai file CSV thành DataFrame
            left_df = read_data(left_df)
            right_df = read_data(right_df)

            if left_df is not None and right_df is not None:
                # Hiển thị dữ liệu của hai DataFrame
                st.write("DataFrame trái:")
                st.write(left_df.head())

                st.write("DataFrame phải:")
                st.write(right_df.head())

                # Chọn phương pháp merge
                merge_method = st.selectbox("Chọn phương pháp merge:", ["left", "right", "outer", "inner"])

                # Chọn cột kết nối (key) cho merge
                merge_on_left = st.selectbox("Chọn cột kết nối (key) cho DataFrame trái (left):", left_df.columns)
                merge_on_right = st.selectbox("Chọn cột kết nối (key) cho DataFrame phải (right):", right_df.columns)

                if st.button("Merge DataFrame"):
                    try:
                        # Merge hai DataFrame dựa vào phương pháp được chọn
                        merged_df = pd.merge(left_df, right_df, how=merge_method, left_on=merge_on_left, right_on=merge_on_right)

                        # Hiển thị kết quả sau khi merge
                        st.success(f"Merge hai DataFrame với phương pháp '{merge_method}' thành công!")
                        st.write("DataFrame sau khi merge:")
                        st.write(merged_df)

                        # Tải xuống DataFrame sau khi merge
                        csv = merged_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="merged_data.csv">Tải xuống CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Có lỗi xảy ra khi merge DataFrame: {e}")