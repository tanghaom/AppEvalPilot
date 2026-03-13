#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/03/09
@File    : selenium_controller.py
@Desc    : Selenium-based browser controller for web automation
"""
import time
from pathlib import Path
from typing import Optional, Tuple
import json
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
try:
    from webdriver_manager.chrome import ChromeDriverManager
    _HAS_WEBDRIVER_MANAGER = True
except ImportError:
    _HAS_WEBDRIVER_MANAGER = False


class SeleniumController:
    """Selenium-based browser controller

    Provides browser automation capabilities using Selenium WebDriver.
    """

    def __init__(
        self,
        debugger_port: int = 9222,
        headless: bool = False,
    ):
        """Initialize Selenium controller

        Args:
            chrome_path: Path to Chrome executable
            debugger_port: Chrome debugger port
            headless: Run in headless mode
        """
        options = Options()
        self.debugger_port = debugger_port
        self.wait_timeout = 10
        self.headless = headless
        self.current_element = None
        if self.headless:
            options.add_argument("--headless")
            options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options)

    def get_window_handles(self):
        window_handles = self.driver.window_handles
        current_window_handle = self.driver.current_window_handle
        return_info = f"全部窗口: {window_handles} \n 当前窗口: {current_window_handle}"
        logger.info(return_info)
        return return_info        

    def get_url(self):
        url = self.driver.current_url
        return_info = f"当前URL: {url}"
        logger.info(return_info)
        return return_info   

    def switch_window_handle(self,step: int):
        try:
            window_handles = self.driver.window_handles
            origin_handle = self.driver.current_window_handle
            idx = window_handles.index(origin_handle)
            if idx+step<=len(window_handles)-1 and idx+step>=0:
                self.driver.switch_to.window(window_handles[idx+step])
                current_handle = self.driver.current_window_handle
                return_info = f"成功切换下一个页面. \n 切换前页面: {origin_handle} \n当前页面 :{current_handle} \n 全部页面: {window_handles}"
                logger.info(return_info)
                return return_info  
            else:
                return_info = f"页面超出范围，无法切换 \n 页面总索引(从0开始): {len(window_handles)} , 当前页面索引: {idx} , 目标索引: {idx+step}"
                logger.error(return_info)
                return return_info       

        except Exception as e:
            return_info = f"页面切换失败: {e}"
            logger.error(return_info)
            return return_info              
    
    def close_window_handle(self):
        try:
            self.driver.close()
            window_handles = self.driver.window_handles
            self.driver.switch_to.window(window_handles[0])
            current_handle = self.driver.current_window_handle
            return_info = f"成功关闭页面. \n当前页面 :{current_handle} \n 全部页面: {window_handles}"
            logger.info(return_info)
            return return_info      
        except Exception as e:
            #current_handle = self.driver.current_window_handle
            window_handles = self.driver.window_handles
            return_info = f"关闭页面失败. \n 全部页面: {window_handles}"
            logger.error(return_info)
            return return_info     
            


    def navigate(self, url: str) -> str:
        try:
            self.driver.get(url)
            time.sleep(3)
            return_info = f"成功跳转URL: {url}"
            logger.info(return_info)
        except Exception as e:
            return_info = f"跳转URL失败: {e}"
            logger.error(return_info)
        finally:
            return return_info	

    def stop(self) -> None:
        """Stop browser session"""
        if self.driver:
            try:
                self.driver.quit()
                return_info = f"浏览器成功关闭。"
            except Exception as e:
                return_info = f"浏览器关闭失败: {e}。"
            finally:
                self.driver = None
                logger.info(return_info)
                
    def report(self,result:str) -> str:
        """report"""
        try:
            dct = json.loads(result)
            assert "task_id" in dct
            assert "task_name" in dct
            assert "result" in dct
            assert "evidence" in dct
            return_info = f"成功提交任务记录: {dct}"
            logger.info(return_info)
        except Exception as e:
            return_info = f"提交任务记录失败: {e}"
            logger.info(return_info)
        finally:
            return return_info

    def get_screenshot(self, filepath: str = "./screenshot/screenshot.png") -> None:
        """Take screenshot

        Args:
            filepath: Path to save screenshot
        """
        if not self.driver:
            logger.error("Browser not started")
            return

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.driver.save_screenshot(filepath)
        logger.info(f"Screenshot saved to {filepath}")

    def click(self, x: int = None, y: int = None) -> None:
        """Click element

        Args:
            x: X coordinate (if using coordinates)
            y: Y coordinate (if using coordinates)
        """
        if not self.driver:
            logger.error("Browser not started")
            return

        try:
            if x is not None and y is not None:
                self.driver.execute_script(f"document.elementFromPoint({x}, {y}).click()")
                logger.info(f"Clicked at ({x}, {y})")
        except Exception as e:
            logger.error(f"Click failed: {e}")

    def type_text(self, text: str) -> None:
        """Type text into element

        Args:
            text: Text to type
            el : target element
        """
        if not self.driver :
            return_info = "浏览器没有运行"
            logger.error(return_info)
            return return_info
        
        elif not self.current_element:
            return_info = "没有选择元素"
            logger.error(return_info)
            return return_info

        try:
            data = extract_element_info(self.current_element)
            self.current_element.clear()
            self.current_element.send_keys(text)
            return_info = f"成功输入文本: {text} \n 输入元素: {data}"
            logger.info(return_info)
            return return_info
        except Exception as e:
            return_info = f"输入文本失败: {e}"
            logger.error(return_info)
            return return_info

    def press_key(self, key: str) -> None:
        """Press keyboard key

        Args:
            key: Key name (e.g., 'enter', 'tab', 'escape')
        """
        if not self.driver:
            logger.error("Browser not started")
            return

        key_map = {
            "enter": Keys.ENTER,
            "tab": Keys.TAB,
            "escape": Keys.ESCAPE,
            "space": Keys.SPACE,
            "backspace": Keys.BACK_SPACE,
        }

        try:
            key_obj = key_map.get(key.lower(), key)
            ActionChains(self.driver).send_keys(key_obj).perform()
            logger.info(f"Pressed key: {key}")
        except Exception as e:
            logger.error(f"Press key failed: {e}")

    def get_page_source(self) -> str:
        """Get page HTML source

        Returns:
            Page HTML source
        """
        if not self.driver:
            logger.error("Browser not started")
            return ""
        return self.driver.page_source

    def click_element(self) -> None:
        data = extract_element_info(self.current_element)
        try:
            self.current_element.click()
            time.sleep(3)
            return_info = f"点击元素：{data} \n 当前url: {self.driver.current_url} \n 当前窗口: {self.driver.window_handles}"
            logger.info(return_info)
            return return_info
        except Exception as e:
            return_info = f"点击元素失败: {e}"
            logger.error(return_info)
            return return_info

    def execute_script(self, script: str) -> any:
        """Execute JavaScript

        Args:
            script: JavaScript code

        Returns:
            Script execution result
        """
        if not self.driver:
            logger.error("Browser not started")
            return None
        return self.driver.execute_script(script)

    def get_window_size(self) -> Tuple[int, int]:
        """Get window size

        Returns:
            (width, height)
        """
        if not self.driver:
            return (0, 0)
        size = self.driver.get_window_size()
        return (size["width"], size["height"])

    def select_element(self, by: str, value: str):
        """Find element by selector

        Args:
            by: Selector type (id, css, class, name, tag)
            value: Selector value

        Returns:
            WebElement or None
        """
        if not self.driver:
            logger.error("Browser not started")
            return None

        by_map = {
            "id": By.ID,
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }

        by_type = by_map.get(by.lower())
        if not by_type:
            logger.error(f"Invalid selector type: {by}")
            return None

        try:
            self.current_element = WebDriverWait(self.driver, self.wait_timeout).until(
                EC.presence_of_element_located((by_type, value))
            )
            data = extract_element_info(self.current_element)
            return_info = f"成功选择元素: {data}"
            logger.info(return_info)
            return return_info

        except Exception as e:
            return_info = f"选择元素失败: {e}"
            logger.error(return_info)
            return return_info

    def find_elements(self, by: str, value: str):
        """Find multiple elements by selector

        Args:
            by: Selector type (id, xpath, css, class, name, tag)
            value: Selector value

        Returns:
            List of WebElements
        """
        if not self.driver:
            logger.error("Browser not started")
            return []

        by_map = {
            "id": By.ID,
            "xpath": By.XPATH,
            "css": By.CSS_SELECTOR,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }

        by_type = by_map.get(by.lower())
        if not by_type:
            logger.info(f"Invalid selector type: {by}")
            return []

        try:
            return self.driver.find_elements(by_type, value)
        except Exception as e:
            logger.error(f"Find elements failed: {e}")
            return []

    def get_core_elements_info(self):
        """Get all core elements on current page

        Returns:
            List of core WebElements (visible and enabled)
        """
        if not self.driver:
            logger.error("Browser not started")
            return []

        try:
            # xpath = """
            #     //*[not(@disabled) and (
            #         self::a or
            #         self::button or
            #         self::input or
            #         self::select or
            #         self::textarea or
            #         @onclick or
            #         @role='button' or
            #         @contenteditable='true'
            #     )]
            # """

            xpath = (
                "//*[("

                # 1. 表单与交互控制
                "self::a or self::button or self::input or self::textarea or self::select or self::option or self::label or self::summary or "

                # 2. 列表与表格内容
                "self::li or self::dt or self::dd or self::td or self::th or self::figcaption or self::caption or self::legend or "
                
                # 3. 独立文本块
                "self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::p or "
                "self::blockquote or self::pre or self::code or self::address or "
                
                # 4. 具有直接游离文本的通用容器
                "((self::div or self::span or self::section or self::article) and text()[normalize-space()]) or "
                
                # 5. 独立媒体与图形
                "self::img or self::video or self::audio or self::iframe or self::svg or self::canvas or "
                  
                # 6. ARIA 交互角色
                "@role='button' or @role='link' or @role='combobox' or @role='tab' or @role='menuitem' or "
                "@role='checkbox' or @role='radio' or @role='switch' or @role='slider' or @role='textbox' or "
                "@role='searchbox' or @role='treeitem' or @role='option' or "
                
                # 7. 兜底交互属性
                "@onclick or @tabindex"
                ") "
                # 排除逻辑 (保持不变)
                "and not(ancestor-or-self::script) "
                "and not(ancestor-or-self::style) "
                "and not(ancestor-or-self::head) "
                "and not(ancestor-or-self::noscript) "
                "and not(ancestor-or-self::template) "
                "and not(ancestor-or-self::meta) "
                "and not(ancestor-or-self::link) "
                "and not(ancestor-or-self::*[@aria-hidden='true']) "
                "and not(ancestor-or-self::*[@hidden]) "
                "and not(ancestor-or-self::*[contains(translate(@style, ' ', ''), 'display:none') or contains(translate(@style, ' ', ''), 'visibility:hidden')])"
                "]"
            )
            elements = self.driver.find_elements(By.XPATH, xpath)



            filter_script = """
                const elements = arguments[0];
                const elementSet = new Set(elements);
                const parentSet = new Set();

                // 定义绝对优先的交互标签（全部小写）
                const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea']);

                for (let el of elements) {
                    let parent = el.parentElement;
                    
                    while (parent) {
                        if (elementSet.has(parent)) {
                            // 如果祖先是一个高优先级的交互元素，当前底层元素就不应该被保留，而是保留那个祖先
                            if (interactiveTags.has(parent.tagName.toLowerCase()) || parent.getAttribute('role') === 'button') {
                                parentSet.add(el); // 丢弃当前的底层元素（比如 a 里面的 span）
                                break; 
                            } else {
                                parentSet.add(parent); // 正常逻辑：丢弃作为容器的祖先（比如 div 里面的 button，丢弃 div）
                            }
                        }
                        parent = parent.parentElement;
                    }
                }

                const result = [];
                for (let el of elements) {
                    if (!parentSet.has(el)) {
                        result.push(el);
                    }
                }
                return result;
            """

            # 执行 JS 过滤，得到真正独立、不重叠的元素列表
            clean_elements = self.driver.execute_script(filter_script, elements)           
            
            core_info = [extract_element_info(el) for el in clean_elements if el.is_displayed() and el.is_enabled()]

            logger.info(f"Found {len(core_info)} core elements")
            return core_info
        except Exception as e:
            logger.error(f"Get core elements failed: {e}")
            return []

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()



def extract_element_info(element):
    """
    提取单个元素的关键信息，包括标签、属性、文本及坐标
    """
    try:
        tag = element.tag_name
        
        # 1. 基础公共信息
        info = {
            "tag_name": tag,
            "text": (element.text or "").strip(),
            "id": element.get_attribute("id") or "",
            "class": element.get_attribute("class") or "",
            "name": element.get_attribute("name") or "",
            "css": element.get_attribute("style") or "",  # 内联样式
        }

        # 2. 针对特殊标签提取特有属性
        if tag == "a":
            #info["href"] = element.get_attribute("href")
            pass
        elif tag in ["input", "button"]:
            info["type"] = element.get_attribute("type")
            info["value"] = element.get_attribute("value")
        elif tag == "img" or element.get_attribute("role") == "img":
            #info["src"] = element.get_attribute("src")
            info["alt"] = element.get_attribute("alt")

        # 3. 兜底策略：获取辅助功能属性
        if not info["text"]:
            info["text"] = (
                element.get_attribute("aria-label") or 
                element.get_attribute("title") or 
                element.get_attribute("placeholder") or 
                ""
            )

        # 4. 获取位置与尺寸信息 (关键新增)
        # .rect 返回一个字典，包含: x, y (坐标) 和 width, height (宽高)
        rect = element.rect
        info.update({
            "x": str(rect.get('x')),
            "y": str(rect.get('y')),
            "width": str(rect.get('width')),
            "height": str(rect.get('height'))
        })

        # 5. 过滤掉值为 None 或空字符串的字段
        # cleaned_info = {k: v for k, v in info.items() if v is not None and v != ""}
        
        return info

    except Exception as e:
        print(f"提取元素信息失败: {e}")
        return {}

    except StaleElementReferenceException:
        # 页面刷新或DOM变化导致元素失效时，跳过该元素
        return {"error": "Stale Element"}
    except Exception as e:
        return {"error": str(e)}