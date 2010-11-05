using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using GASS.OpenCL;

namespace OpenCLTemplate
{
    /// <summary>Displays OpenCL related information</summary>
    public partial class frmCLInfo : Form
    {
        private void frmCLInfo_Load(object sender, EventArgs e)
        {
            CLCalc.InitCL();
            if (CLCalc.CLAcceleration != CLCalc.CLAccelerationType.UsingCL)
            {
                cmbPlat.Items.Add("OpenCL ERROR");
                if (cmbPlat.Items.Count > 0) cmbPlat.SelectedIndex = 0;
            }
            else
            {
                foreach(CLCalc.CLPlatform p in CLCalc.CLPlatforms)
                {
                    cmbPlat.Items.Add(p.CLPlatformName + " " + p.CLPlatformProfile + " " + p.CLPlatformVendor + " " + p.CLPlatformVersion);
                }
                if (cmbPlat.Items.Count > 0) cmbPlat.SelectedIndex = 0;

                int i=0;
                foreach (CLCalc.CLDevice d in CLCalc.CLDevices)
                {
                    if (d.CLDeviceAvailable)
                    {
                        cmbDevices.Items.Add(d.CLDeviceName + " " + d.CLDeviceType + " " + d.CLDeviceVendor + " " + d.CLDeviceVersion);
                        cmbCurDevice.Items.Add(d.CLDeviceName + " " + d.CLDeviceType + " " + d.CLDeviceVendor + " " + d.CLDeviceVersion);
                    }
                    else
                    {
                        cmbDevices.Items.Add("NOT AVAILABLE: " + d.CLDeviceName + " " + d.CLDeviceType + " " + d.CLDeviceVendor + " " + d.CLDeviceVersion);
                        cmbCurDevice.Items.Add("NOT AVAILABLE: " + d.CLDeviceName + " " + d.CLDeviceType + " " + d.CLDeviceVendor + " " + d.CLDeviceVersion);
                    }

                    i++;
                }

                if (cmbDevices.Items.Count > 0)
                {
                    cmbDevices.SelectedIndex = 0;
                    cmbCurDevice.SelectedIndex = CLCalc.Program.DefaultCQ;
                }
            }

            ReadImportantRegistryEntries();
        }

        private void ReadImportantRegistryEntries()
        {
            //Reads registry keys
            Utility.ModifyRegistry.ModifyRegistry reg = new Utility.ModifyRegistry.ModifyRegistry();
            reg.SubKey = "SYSTEM\\CURRENTCONTROLSET\\CONTROL\\SESSION MANAGER\\ENVIRONMENT";
            try
            {
                string s = (string)reg.Read("GPU_MAX_HEAP_SIZE");
                lblGPUHeap.Text = s == null ? lblNotFound.Text : s;
            }
            catch
            {
                lblGPUHeap.Text = lblNotFound.Text;
            }

            reg.SubKey = "SYSTEM\\CURRENTCONTROLSET\\CONTROL\\GraphicsDrivers";
            int val;
            try
            {
                val = (int)reg.Read("TdrDelay");
                lblTdrDelay.Text = val.ToString();
            }
            catch
            {
                lblTdrDelay.Text = lblNotFound.Text;
            }

            try
            {
                val = (int)reg.Read("TdrDdiDelay");
                lblTdrDdiDelay.Text = val.ToString();
            }
            catch
            {
                lblTdrDdiDelay.Text = lblNotFound.Text;
            }

            ulong size = 32;
            for (int i = 0; i < CLCalc.CLDevices.Count; i++)
            {
                if (CLCalc.CLDevices[i].CLDeviceType == "4" || CLCalc.CLDevices[i].CLDeviceType == "8")
                {
                    if (CLCalc.CLDevices[i].CLDeviceMemSize > size)
                        size = CLCalc.CLDevices[i].CLDeviceMemSize / (1024 * 1024);
                }
            }

            lblRecomHeapSize.Text = "90";// size.ToString();
            lblRecomTdrDdiDelay.Text = "256";
            lblRecomTdrDelay.Text = "128";

        }

        private void btnWriteToRegistry_Click(object sender, EventArgs e)
        {
            string msg = lblConfirmModReg.Text + "\n";
            msg += "HKEY_LOCAL_MACHINE\\SYSTEM\\CURRENTCONTROLSET\\CONTROL\\SESSION MANAGER\\ENVIRONMENT - GPU_MAX_HEAP_SIZE = " + lblRecomHeapSize.Text + "\n";
            msg += "HKEY_LOCAL_MACHINE\\SYSTEM\\CURRENTCONTROLSET\\CONTROL\\GraphicsDrivers - TdrDelay = " + lblTdrDelay.Text + "\n";
            msg += "HKEY_LOCAL_MACHINE\\SYSTEM\\CURRENTCONTROLSET\\CONTROL\\GraphicsDrivers - TdrDdiDelay = " + lblTdrDdiDelay.Text + "\n";

            if (MessageBox.Show(msg, this.Text, MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
            {
                Utility.ModifyRegistry.ModifyRegistry reg = new Utility.ModifyRegistry.ModifyRegistry();
                reg.SubKey = "SYSTEM\\CURRENTCONTROLSET\\CONTROL\\SESSION MANAGER\\ENVIRONMENT";
                try
                {
                    reg.Write("GPU_MAX_HEAP_SIZE", lblRecomHeapSize.Text);
                }
                catch { }

                reg.SubKey = "SYSTEM\\CURRENTCONTROLSET\\CONTROL\\GraphicsDrivers";
                int val;
                try
                {
                    int.TryParse(lblRecomTdrDelay.Text, out val);
                    reg.Write("TdrDelay", val);
                }
                catch { }

                try
                {
                    int.TryParse(lblRecomTdrDdiDelay.Text, out val);
                    reg.Write("TdrDdiDelay", val);
                }
                catch { }

                ReadImportantRegistryEntries();

                MessageBox.Show(lblReboot.Text, this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private void cmbDevices_SelectedIndexChanged(object sender, EventArgs e)
        {
            int ind = cmbDevices.SelectedIndex;
            lstDevDetails.Items.Clear();
            CLCalc.CLDevice d = CLCalc.CLDevices[ind];
            lstDevDetails.Items.Add("Name: " + d.CLDeviceName);
            lstDevDetails.Items.Add("Type: " + d.CLDeviceType);
            lstDevDetails.Items.Add("Vendor: " + d.CLDeviceVendor);
            lstDevDetails.Items.Add("Version: " + d.CLDeviceVersion);
            lstDevDetails.Items.Add("Memory size (Mb): " + d.CLDeviceMemSize/(1024*1024));
            lstDevDetails.Items.Add("Maximum allocation size (Mb):" + d.CLDeviceMaxMallocSize / (1024 * 1024));
            lstDevDetails.Items.Add("Compiler available? " + d.CLDeviceCompilerAvailable);
            lstDevDetails.Items.Add("Device available? " + d.CLDeviceAvailable);
        }


        /// <summary>Constructor.</summary>
        public frmCLInfo()
        {
            System.Threading.Thread.CurrentThread.CurrentUICulture = new System.Globalization.CultureInfo(System.Globalization.CultureInfo.CurrentCulture.LCID);
            InitializeComponent();
        }

        private void frmCLInfo_DoubleClick(object sender, EventArgs e)
        {
            //frmCLEdit frmEdit = new frmCLEdit();
            //frmEdit.ShowDialog();
        }

        private void cmbCurDevice_SelectedIndexChanged(object sender, EventArgs e)
        {
            CLCalc.Program.DefaultCQ = cmbCurDevice.SelectedIndex;
        }



    }
}
