"""Installed Edge end-to-end acceptance. Real standalone Train-derived upload."""
import json,time,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from .registry import OUT
from .schema import validate_prediction

EDGE="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
def main():
    valid="--valid-fixture" in sys.argv
    begin=time.perf_counter();out=OUT/("acceptance/browser_valid" if valid else "acceptance/browser");out.mkdir(exist_ok=True)
    fixture=OUT/("acceptance/fixtures/real_valid_upload.mp4" if valid else "acceptance/fixtures/real_train_upload.mp4")
    checks=[];errors=[];report=dict(Test_read=False,source="independent uploaded file from Valid (not trained on)" if valid else "independent uploaded file from Train", external_unseen=False)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=EDGE,headless=True,args=["--no-first-run","--disable-background-networking"])
        page=browser.new_page(viewport={"width":1500,"height":1050})
        page.on("pageerror",lambda e:errors.append(str(e)))
        try:
            page.goto("http://127.0.0.1:8502",wait_until="networkidle")
            expect(page.get_by_role("heading",name="AnimalAct 2.0 Recognition",exact=True)).to_be_visible(timeout=60000)
            expect(page.get_by_role("button",name="Run Recognition",exact=True)).to_be_disabled()
            checks.append("Both servers and initial Video page are reachable")
            page.locator('input[type="file"]').set_input_files(str(fixture))
            expect(page.get_by_role("button",name="Run Recognition",exact=True)).to_be_enabled(timeout=30000)
            page.wait_for_function("document.querySelector('video') && document.querySelector('video').readyState>=2",timeout=30000)
            page.locator("video").first.evaluate("async v => {v.muted=true; await v.play();}")
            page.wait_for_function("document.querySelector('video').currentTime > 0.1")
            checks.append("Real uploaded video previews and plays in Edge")
            page.get_by_role("button",name="Run Recognition",exact=True).click()
            expect(page.get_by_role("button",name="Recognition complete",exact=True)).to_be_visible(timeout=180000)
            page.screenshot(path=str(out/"video_prediction.png"),full_page=True)
            page.get_by_role("button",name="View detailed results →").click()
            expect(page.get_by_role("heading",name="Recognition Results",exact=True)).to_be_visible()
            with page.expect_download() as item: page.get_by_role("button",name="Download JSON",exact=True).click()
            item.value.save_as(str(out/"attention_prediction.json"))
            result=validate_prediction(json.loads((out/"attention_prediction.json").read_text(encoding="utf-8")))
            assert result["model_id"]=="attention" and result["threshold"]==.12
            assert result["feature_shape"]==[1,8,768] and (valid or len(result["predicted_labels"])>=2)
            for name in result["predicted_labels"]:expect(page.locator(".action-tags").get_by_text(name,exact=True)).to_be_visible()
            with page.expect_download() as item:page.get_by_role("button",name="Download CSV",exact=True).click()
            item.value.save_as(str(out/"prediction.csv"))
            display=page.get_by_role("combobox",name="Display",exact=True)
            display.focus();display.press("ArrowDown");display.press("End");display.press("Enter")
            expect(page.get_by_text("Showing 119 of 119 class scores.",exact=False)).to_be_visible()
            page.screenshot(path=str(out/"results.png"),full_page=True)
            checks.append("Actual Attention inference displays all selected labels, all119 scores and real CSV/JSON exports")
            page.get_by_test_id("stSidebar").get_by_text("Evaluation",exact=True).click()
            expect(page.get_by_role("heading",name="AnimalAct 2.0 Evaluation",exact=True)).to_be_visible()
            for tab in ("Overview","Per-class AP","Confusion","Training","Generalization"):
                page.get_by_role("tab",name=tab,exact=True).click()
                expect(page.get_by_role("tab",name=tab,exact=True)).to_have_attribute("aria-selected","true")
                expect(page.get_by_test_id("stException")).to_have_count(0)
                page.screenshot(path=str(out/(tab.replace(" ","_")+".png")),full_page=True)
            checks.append("All five Evaluation tabs render real experiment data without inference or Test results")
            page.get_by_test_id("stSidebar").get_by_text("Video",exact=True).click()
            head=page.get_by_role("combobox",name="Classification head",exact=True)
            head.focus();head.press("ArrowDown");head.press("End");head.press("Enter")
            expect(page.get_by_role("button",name="Run Recognition",exact=True)).to_be_enabled()
            expect(page.get_by_role("heading",name="Recognition preview",exact=True)).to_have_count(0)
            page.get_by_role("button",name="Run Recognition",exact=True).click()
            expect(page.get_by_role("button",name="Recognition complete",exact=True)).to_be_visible(timeout=120000)
            page.get_by_role("button",name="View detailed results →").click()
            with page.expect_download() as item:page.get_by_role("button",name="Download JSON",exact=True).click()
            item.value.save_as(str(out/"bigru_mean_prediction.json"))
            other=validate_prediction(json.loads((out/"bigru_mean_prediction.json").read_text(encoding="utf-8")))
            assert other["model_id"]=="bigru_mean" and other["threshold"]==.07
            checks.append("Switching head invalidates stale prediction and executes real BiGRU Mean checkpoint")
            page.get_by_test_id("stSidebar").get_by_text("Video",exact=True).click()
            page.get_by_text("Select a clip window",exact=True).click()
            end=page.get_by_role("spinbutton",name="Window end (seconds)",exact=True);end.fill(str(2.5/result["video"]["fps"]));end.press("Enter")
            expect(page.get_by_role("button",name="Run Recognition",exact=True)).to_be_enabled()
            page.get_by_role("button",name="Run Recognition",exact=True).click()
            expect(page.get_by_role("button",name="Recognition complete",exact=True)).to_be_visible(timeout=120000)
            page.get_by_role("button",name="View detailed results →").click()
            with page.expect_download() as item:page.get_by_role("button",name="Download JSON",exact=True).click()
            item.value.save_as(str(out/"short_window_prediction.json"))
            short=validate_prediction(json.loads((out/"short_window_prediction.json").read_text(encoding="utf-8")))
            assert short["video"]["real_length"]==3 and short["feature_shape"]==[1,2,768]
            assert short["video"]["sampled_frame_indices"]==[0,1,2]
            checks.append("Three-frame odd window uses every real frame once and returns two valid tubelets")
            page.get_by_test_id("stSidebar").get_by_text("Video",exact=True).click()
            corrupt=OUT/"acceptance/fixtures/corrupt.mp4";corrupt.write_bytes(b"not a real video")
            page.locator('input[type="file"]').set_input_files(str(corrupt))
            expect(page.get_by_role("button",name="Run Recognition",exact=True)).to_be_disabled(timeout=30000)
            expect(page.get_by_role("heading",name="Recognition preview",exact=True)).to_have_count(0)
            expect(page.get_by_test_id("stException")).to_have_count(0)
            checks.append("Invalid replacement upload clears old result and disables inference")
            assert not errors,errors
            report.update(passed=True,labels=result["predicted_labels"],inference_seconds=result["inference_time"],checks=checks)
        except Exception as exc:
            page.screenshot(path=str(out/"failure.png"),full_page=True)
            (out/"failure_dom.txt").write_text(page.locator("body").inner_text(),encoding="utf-8")
            report.update(passed=False,error=repr(exc),checks=checks)
        finally:
            report.update(page_errors=errors,wall_seconds=time.perf_counter()-begin)
            (out/"acceptance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps(report,ensure_ascii=False),flush=True);browser.close()
    return report["passed"]
if __name__=="__main__":raise SystemExit(0 if main() else 1)
