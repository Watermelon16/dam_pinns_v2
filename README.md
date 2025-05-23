# Công cụ tính toán tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs

## Giới thiệu

Đây là ứng dụng tính toán tối ưu mặt cắt kinh tế của đập bê tông trọng lực sử dụng mô hình Physics-Informed Neural Networks (PINNs). Ứng dụng này kết hợp kiến thức vật lý với cơ học đập và sức mạnh của học sâu để tìm ra mặt cắt tối ưu nhất, đảm bảo các điều kiện ổn định và an toàn.

## Tính năng

- **Tính toán tối ưu sử dụng PINNs**: Áp dụng mô hình mạng nơ-ron học sâu kết hợp với các ràng buộc vật lý
- **Giao diện người dùng**: Thiết kế đơn giản, theo phong cách hiện đại
- **Trực quan hóa**: Hiển thị đồ thị lực tác dụng và biểu đồ hàm mất mát
- **Báo cáo**: Xuất báo cáo dạng Excel
- **Cơ sở dữ liệu**: Lưu trữ và quản lý lịch sử tính toán
- **Triển khai miễn phí**: Tương thích với Streamlit Cloud

## Truy cập ứng dụng

Ứng dụng có thể được triển khai trực tuyến bằng cách làm theo hướng dẫn trong phần "Triển khai" bên dưới.

## Cài đặt cục bộ

### Yêu cầu

- Python 3.9 trở lên
- PyTorch 1.13.1
- Các thư viện được liệt kê trong `requirements.txt`

### Cài đặt

```bash
git clone https://github.com/yourusername/dam-optimizer-pinns.git
cd dam-optimizer-pinns
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc thư mục

```
dam_optimizer_pinns/
├── app.py              # Ứng dụng chính
├── database.py         # Quản lý lịch sử tính toán
└── requirements.txt    # Danh sách thư viện
```

## Cách sử dụng

1. Nhập các thông số đầu vào (chiều cao đập, trọng lượng riêng, hệ số ma sát, v.v.)
2. Nhấn nút “Tính toán tối ưu”
3. Xem kết quả tối ưu hóa
4. Xuất báo cáo hoặc lưu lịch sử

## Lý thuyết

### Physics-Informed Neural Networks (PINNs)

PINNs là một phương pháp kết hợp dữ liệu với các ràng buộc vật lý. Trong ứng dụng này, PINNs được sử dụng để tìm các tham số tối ưu của mặt cắt đập bê tông trọng lực, đảm bảo các điều kiện:

- Ổn định trượt: Hệ số ổn định `K ≥ Kc`
- Không có ứng suất kéo: Ứng suất mép thượng lưu `σ ≤ 0`
- **Tối thiểu hóa diện tích mặt cắt**: Giảm thiểu lượng bê tông sử dụng

## Giấy phép

Sử dụng theo MIT License.
