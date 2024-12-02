from vietnamese_summarizer import VietnameseSummarizer

def test_summarizer():
    # Initialize the summarizer
    summarizer = VietnameseSummarizer(device="cpu")
    
    # Input Vietnamese text
    input_text = """
    Lần đầu tiên Mike Tyson bước lên võ đài. Vẻ ngoài trẻ trung của ông phản ảnh sức mạnh mà ông sở hữu. Năm 1985 ông bước vào lãnh vực chuyên nghiệp của một võ sĩ quyền anh hạng nặng.

    Lúc đó, chỉ trong 1 phút 47 giây sau, Tyson, 18 tuổi, đã buộc đối thủ Hector Mercedes phải rút lui sau hàng loạt cú đấm tàn khốc vào người và đầu, theo CNN Sports ngày 14 Tháng Mười Một.Đó là sự khởi đầu cho một sự nghiệp đáng gờm trên võ đài của Tyson, với đầy những thăng trầm “lên voi xuống chó.” Giờ đây, 39 năm sau, “Iron Mike” sẽ trở lại sàn đấu ở tuổi 58.

    Vào Thứ Sáu, 15 Tháng Mười Một, Tyson đeo găng tay một lần nữa trong trận đấu chuyên nghiệp với YouTuber Jake Paul, 27 tuổi, tại AT&T Stadium ở Arlington, Texas, sau hơn 7,000 ngày kể từ trận đấu chuyên nghiệp cuối cùng của ông.

    Trận đấu chuyên nghiệp cuối cùng của Tyson đã thua Kevin McBride hơn 19 năm trước, và trận đấu “biểu diễn” cuối cùng của Tyson với Paul là bốn năm trước.

    Hai tay quyền anh này ban đầu dự kiến thi đấu vào Tháng Bảy năm nay, tuy nhiên trận đấu đã bị hoãn lại khi Tyson bị bệnh loét bao tử tái phát.

    Tyson đã mang tất cả những “rủi ro” về sức khỏe vào sự nghiệp quyền anh lâu dài của ông và thường gây tranh cãi.
    """
    
    # Generate the summary
    result = summarizer.generate_summary(input_text, translate_to_english=True)
    
    # Print the results
    print("Vietnamese Summary:")
    print(result['vietnamese_summary'])
    print("\nEnglish Summary:")
    print(result['english_summary'])

if __name__ == "__main__":
    test_summarizer()
