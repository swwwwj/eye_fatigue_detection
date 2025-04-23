import pandas as pd
from pathlib import Path
import requests
import json
from datetime import datetime

class FatigueDataAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.moonshot.cn/v1/chat/completions"  # 替换为实际的 API 地址
        self.data_dir = Path(r"D:\github\eye_fatigue_detection\face_data")

    def _read_data(self, date):
        """读取指定日期的疲劳检测数据"""
        try:
            filename = self.data_dir / f"fatigue_data_{date}.csv"
            if not filename.exists():
                return None
            return pd.read_csv(filename)
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None

    def _generate_report_prompt(self, df):
        """生成用于 Deepseek 的提示"""
        summary_stats = {
            'awake': len(df[df['label'] == 'awake']),
            'mild_fatigue': len(df[df['label'] == 'mild_fatigue']),
            'moderate_fatigue': len(df[df['label'] == 'moderate_fatigue']),
            'severe_fatigue': len(df[df['label'] == 'severe_fatigue'])
        }
        
        prompt = f"""
请根据以下疲劳检测数据生成分析报告：

数据时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}

各状态检测次数：
- 清醒: {summary_stats['awake']} 次
- 轻度疲劳: {summary_stats['mild_fatigue']} 次
- 中度疲劳: {summary_stats['moderate_fatigue']} 次
- 重度疲劳: {summary_stats['severe_fatigue']} 次

请分析以下几个方面：
1. 用户的整体疲劳状况
2. 疲劳程度的变化趋势
3. 对用户的建议
4. 潜在的健康风险提示

请用中文生成一份专业的分析报告。
"""
        return prompt

    def _call_Kimi_api(self, prompt):
        """调用 Kimi API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "moonshot-v1-8k",  
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的疲劳分析师，需要基于数据生成专业的分析报告。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)

            if response.status_code != 200:
                print(f"API 请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return f"API 请求失败: HTTP {response.status_code}"

            try:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print(f"API 响应格式异常: {result}")
                    return "API 响应格式异常"
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}")
                print(f"响应内容: {response.text}")
                return "API 响应解析失败"
                
        except requests.RequestException as e:
            print(f"API 请求异常: {e}")
            return f"API 请求异常: {str(e)}"
        except Exception as e:
            print(f"未知错误: {e}")
            return f"生成报告时发生错误: {str(e)}"

    def generate_analysis_report(self, date_str):
        """生成指定日期的分析报告"""
        try:
            date = datetime.strptime(date_str, '%Y%m%d')
            
            df = self._read_data(date_str)
            if df is None:
                return {
                    'status': 'error',
                    'message': '无法读取指定日期的数据'
                }
            
            prompt = self._generate_report_prompt(df)
            report = self._call_Kimi_api(prompt)
            
            if report is None:
                return {
                    'status': 'error',
                    'message': '生成报告失败'
                }
            
            return {
                'status': 'success',
                'report': report,
                'date': date_str
            }
            
        except ValueError:
            return {
                'status': 'error',
                'message': '日期格式无效'
            }

    def export_report(self, date_str, report_content, format='txt'):
        """导出分析报告，支持 txt 和 html 格式"""
        try:
            if format == 'txt':
                output_path = self.data_dir / f"fatigue_report_{date_str}.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"疲劳检测分析报告\n")
                    f.write(f"生成日期：{date_str}\n")
                    f.write("="*50 + "\n\n")
                    f.write(report_content)
                    
            elif format == 'html':
                output_path = self.data_dir / f"fatigue_report_{date_str}.html"
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 40px; }}
                        h1 {{ color: #333; }}
                        .report {{ line-height: 1.6; }}
                    </style>
                </head>
                <body>
                    <h1>疲劳检测分析报告</h1>
                    <p>生成日期：{date_str}</p>
                    <hr>
                    <div class="report">
                        {report_content.replace('\n', '<br>')}
                    </div>
                </body>
                </html>
                """
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            return {
                'status': 'success',
                'message': f'报告已成功导出为{format.upper()}格式',
                'file_path': str(output_path)
            }
            
        except Exception as e:
            print(f"导出报告时出错: {e}")
            return {
                'status': 'error',
                'message': f'导出报告失败: {str(e)}'
            }
        
if __name__ == "__main__":
    analyzer = FatigueDataAnalyzer("your_api_key_here")
    result = analyzer.generate_analysis_report("20250413")
    print(result)